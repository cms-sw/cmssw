// -*- C++ -*-
//
// Package:    HLTLogMonitorFilter
// Class:      HLTLogMonitorFilter
//
/**\class HLTLogMonitorFilter HLTLogMonitorFilter.cc Work/HLTLogMonitorFilter/src/HLTLogMonitorFilter.cc

 Description: Accept events if any LogError or LogWarning was raised

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Bocci
//         Created:  Thu Nov  5 15:16:46 CET 2009
//

// system include files
#include <cstdint>
#include <oneapi/tbb/concurrent_unordered_map.h>
#include <set>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Common/interface/ErrorSummaryEntry.h"
#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"

//
// class declaration
//

class HLTLogMonitorFilter : public edm::global::EDFilter<> {
public:
  explicit HLTLogMonitorFilter(const edm::ParameterSet&);
  ~HLTLogMonitorFilter() override;

  struct CategoryEntry {
    uint32_t
        threshold;  // configurable threshold, after which messages in this Category start to be logarithmically prescaled
    std::atomic<uint32_t> maxPrescale;  // maximum prescale used for this Category
    std::atomic<uint64_t> counter;      // number of events that fired this Category
    std::atomic<uint64_t> accepted;     // number of events acepted for this Category
    uint32_t id;                        // monotonically increasing unique id for this Category

    CategoryEntry(uint32_t iID = 0, uint32_t t = 0)
        : threshold(
              t),  // default-constructed entries have the threshold set to 0, which means the associated Category is disabled
          maxPrescale(1),
          counter(0),
          accepted(0),
          id(iID) {}

    CategoryEntry(CategoryEntry const& iOther)
        : threshold(iOther.threshold),
          maxPrescale(iOther.maxPrescale.load()),
          counter(iOther.counter.load()),
          accepted(iOther.counter.load()),
          id(iOther.id) {}

    // caller guarantees this is called just once per event
    bool accept() {
      auto tCounter = ++counter;

      // fail if the prescaler is disabled (threshold == 0), or if the counter is not a multiple of the prescale
      if (threshold == 0) {
        return false;
      }

      if (threshold == 1) {
        ++accepted;
        return true;
      }

      uint64_t dynPrescale = 1;
      // quasi-logarithmic increase in the prescale factor
      // dynamically calculating the prescale is mulit-thread stable
      while (tCounter > dynPrescale * threshold) {
        dynPrescale *= threshold;
      }

      auto tMaxPrescale = maxPrescale.load();
      while (tMaxPrescale < dynPrescale) {
        maxPrescale.compare_exchange_strong(tMaxPrescale, dynPrescale);
      }

      if (0 != tCounter % dynPrescale) {
        return false;
      }

      ++accepted;
      return true;
    }
  };

  typedef tbb::concurrent_unordered_map<std::string, CategoryEntry> CategoryMap;

  // ---------- private methods -----------------------

  /// EDFilter accept method
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  /// EDFilter beginJob method
  void beginJob() override;

  /// EDFilter endJob method
  void endJob() override;

  /// create a new entry for the given category, with the given threshold value
  CategoryEntry& addCategory(const std::string& category, uint32_t threshold);

  /// return the entry for requested category, if it exists, or create a new one with the default threshold value
  CategoryEntry& getCategory(const std::string& category) const;

  /// summarize to LogInfo
  void summary() const;

  // ---------- member data ---------------------------
  uint32_t m_prescale;  // default threshold, after which messages in each Category start to be logarithmically prescaled
  mutable std::atomic<uint32_t> m_nextEntryID = 0;
  CMS_THREAD_SAFE mutable CategoryMap m_data;  // map each category name to its prescale data
  edm::EDPutTokenT<std::vector<edm::ErrorSummaryEntry>> m_putToken;
};

// system include files
#include <sstream>
#include <iomanip>
#include <memory>
#include <boost/range.hpp>
#include <boost/algorithm/string.hpp>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageSender.h"

//
// constructors and destructor
//
HLTLogMonitorFilter::HLTLogMonitorFilter(const edm::ParameterSet& config) : m_prescale(), m_data() {
  m_prescale = config.getParameter<uint32_t>("default_threshold");

  typedef std::vector<edm::ParameterSet> VPSet;
  const VPSet& categories = config.getParameter<VPSet>("categories");
  for (auto const& categorie : categories) {
    const std::string& name = categorie.getParameter<std::string>("name");
    uint32_t threshold = categorie.getParameter<uint32_t>("threshold");
    addCategory(name, threshold);
  }

  m_putToken = produces<std::vector<edm::ErrorSummaryEntry>>();
}

HLTLogMonitorFilter::~HLTLogMonitorFilter() = default;

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTLogMonitorFilter::filter(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  // no LogErrors or LogWarnings, skip processing and reject the event
  if (not edm::FreshErrorsExist(event.streamID().value()))
    return false;

  //keep track of which errors have already been seen this event
  struct Cache {
    Cache() : done(false), cachedValue(false) {}
    bool done;
    bool cachedValue;
  };
  std::vector<Cache> doneCache(m_nextEntryID);

  // look at the LogErrors and LogWarnings, and accept the event if at least one is under threshold or matches the prescale
  bool accept = false;
  std::string category;

  std::vector<edm::messagelogger::ErrorSummaryEntry> errorSummary{edm::LoggedErrorsSummary(event.streamID().value())};
  for (auto const& entry : errorSummary) {
    // split the message category
    typedef boost::split_iterator<std::string::const_iterator> splitter;
    for (splitter i = boost::make_split_iterator(entry.category, boost::first_finder("|", boost::is_equal()));
         i != splitter();
         ++i) {
      // extract the substring corresponding to the split_iterator
      // FIXME: this can be avoided if the m_data map is keyed on boost::sub_range<std::string>
      category.assign(i->begin(), i->end());

      // access the message category, or create a new one as needed, and check the prescale
      auto& cat = getCategory(category);
      if (cat.id >= doneCache.size()) {
        //new categories were added so need to grow
        doneCache.resize(cat.id + 1);
      }
      if (not doneCache[cat.id].done) {
        doneCache[cat.id].cachedValue = cat.accept();
        doneCache[cat.id].done = true;
      }
      if (doneCache[cat.id].cachedValue)
        accept = true;
    }
  }

  // harvest the errors, but only if the filter will accept the event
  std::vector<edm::ErrorSummaryEntry> errors;
  if (accept) {
    errors.reserve(errorSummary.size());
    std::transform(errorSummary.begin(), errorSummary.end(), std::back_inserter(errors), [](auto& iEntry) {
      edm::ErrorSummaryEntry entry;
      entry.category = std::move(iEntry.category);
      entry.module = std::move(iEntry.module);
      entry.severity = edm::ELseverityLevel(iEntry.severity.getLevel());
      entry.count = iEntry.count;
      return entry;
    });
  }
  event.emplace(m_putToken, std::move(errors));

  return accept;
}

// ------------ method called at the end of the Job ---------
void HLTLogMonitorFilter::beginJob() { edm::EnableLoggedErrorsSummary(); }
// ------------ method called at the end of the Job ---------
void HLTLogMonitorFilter::endJob() {
  edm::DisableLoggedErrorsSummary();
  summary();
}

/// create a new entry for the given category, with the given threshold value
HLTLogMonitorFilter::CategoryEntry& HLTLogMonitorFilter::addCategory(const std::string& category, uint32_t threshold) {
  // check after inserting, as either the new CategoryEntry is needed, or an error condition is raised
  auto id = m_nextEntryID++;
  std::pair<CategoryMap::iterator, bool> result = m_data.insert(std::make_pair(category, CategoryEntry(id, threshold)));
  if (not result.second)
    throw cms::Exception("Configuration") << "Duplicate entry for category " << category;
  return result.first->second;
}

/// return the entry for requested category, if it exists, or create a new one with the default threshold value
HLTLogMonitorFilter::CategoryEntry& HLTLogMonitorFilter::getCategory(const std::string& category) const {
  // check before inserting, to avoid the construction of a CategoryEntry object
  auto i = m_data.find(category);
  if (i != m_data.end())
    return i->second;
  else {
    auto id = m_nextEntryID++;
    return m_data.insert(std::make_pair(category, CategoryEntry(id, m_prescale))).first->second;
  }
}

/// summarize to LogInfo
void HLTLogMonitorFilter::summary() const {
  std::stringstream out;
  out << "Log-Report ---------- HLTLogMonitorFilter Summary ------------\n"
      << "Log-Report  Threshold   Prescale     Issued   Accepted   Rejected Category\n";

  std::set<std::string> sortedCategories;
  for (auto const& entry : m_data) {
    sortedCategories.insert(entry.first);
  }

  for (auto const& cat : sortedCategories) {
    auto entry = m_data.find(cat);
    out << "Log-Report " << std::right << std::setw(10) << entry->second.threshold << ' ' << std::setw(10)
        << entry->second.maxPrescale << ' ' << std::setw(10) << entry->second.counter << ' ' << std::setw(10)
        << entry->second.accepted << ' ' << std::setw(10) << (entry->second.counter - entry->second.accepted) << ' '
        << std::left << cat << '\n';
  }
  edm::LogVerbatim("Report") << out.str();
}

// define as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTLogMonitorFilter);
