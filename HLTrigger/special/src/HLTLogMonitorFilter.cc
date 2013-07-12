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
// $Id: HLTLogMonitorFilter.cc,v 1.9 2012/01/21 15:00:21 fwyzard Exp $
//


// system include files
#include <stdint.h>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Provenance/interface/EventID.h"

//
// class declaration
//

class HLTLogMonitorFilter : public edm::EDFilter {
public:
    explicit HLTLogMonitorFilter(const edm::ParameterSet &);
    ~HLTLogMonitorFilter();

private:
    // ---------- private data types --------------------

    struct CategoryEntry {
      uint32_t threshold;       // configurable threshold, after which messages in this Category start to be logarithmically prescaled
      uint32_t prescale;        // current prescale for this Category
      uint64_t counter;         // number of events that fired this Category
      uint64_t accepted;        // number of events acepted for this Category
      bool     done;            // track if this Category has already been seen in the current event

      CategoryEntry(uint32_t t = 0) :
        threshold(t),           // default-constructed entries have the threshold set to 0, which means the associated Category is disabled
        prescale(1),
        counter(0),
        accepted(0),
        done(false)
      { }

      bool accept() {
        if (not done) {
          done = true;
          ++counter;
        }

        // fail if the prescaler is disabled (threshold == 0), or if the counter is not a multiple of the prescale
        if ((threshold == 0) or (counter % prescale))
          return false;

        // quasi-logarithmic increase in the prescale factor (should be safe even if threshold is 1)
        if (counter == prescale * threshold)
          prescale *= threshold;

        ++accepted;
        return true;
      }

    };

    typedef std::map<std::string, CategoryEntry> CategoryMap;

    // ---------- private methods -----------------------

    /// EDFilter accept method
    virtual bool filter(edm::Event&, const edm::EventSetup &);

    /// EDFilter beginJob method
    virtual void beginJob(void);

    /// EDFilter endJob method
    virtual void endJob(void);

    /// check if the requested category has a valid entry
    bool knownCategory(const std::string & category);

    /// create a new entry for the given category, with the given threshold value
    CategoryEntry & addCategory(const std::string & category, uint32_t threshold);

    /// return the entry for requested category, if it exists, or create a new one with the default threshold value
    CategoryEntry & getCategory(const std::string & category);

    /// summarize to LogInfo
    void summary(void);


    // ---------- member data ---------------------------
    uint32_t    m_prescale;    // default threshold, after which messages in each Category start to be logarithmically prescaled
    CategoryMap m_data;        // map each category name to its prescale data
};


// system include files
#include <sstream>
#include <iomanip>
#include <memory>
#include <boost/foreach.hpp>
#include <boost/range.hpp>
#include <boost/algorithm/string.hpp>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageSender.h"

//
// constructors and destructor
//
HLTLogMonitorFilter::HLTLogMonitorFilter(const edm::ParameterSet & config) :
  m_prescale(),
  m_data()
{
  m_prescale = config.getParameter<uint32_t>("default_threshold");

  typedef std::vector<edm::ParameterSet> VPSet; 
  const VPSet & categories = config.getParameter<VPSet>("categories");
  for (VPSet::const_iterator category = categories.begin(); category != categories.end(); ++category) {
    const std::string & name = category->getParameter<std::string>("name");
    uint32_t threshold       = category->getParameter<uint32_t>("threshold");
    addCategory(name, threshold);
  }

  produces<std::vector<edm::ErrorSummaryEntry> >();
}

HLTLogMonitorFilter::~HLTLogMonitorFilter()
{
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTLogMonitorFilter::filter(edm::Event & event, const edm::EventSetup & setup) {
  // no LogErrors or LogWarnings, skip processing and reject the event
  if (not edm::MessageSender::freshError)
    return false;

  // clear "done" flag in all Categories
  BOOST_FOREACH(CategoryMap::value_type & entry, m_data)
    entry.second.done = false;

  // look at the LogErrors and LogWarnings, and accept the event if at least one is under threshold or matches the prescale
  bool accept = false;
  std::string category;

  typedef std::map<edm::ErrorSummaryMapKey, unsigned int> ErrorSummaryMap;
  BOOST_FOREACH(const ErrorSummaryMap::value_type & entry, edm::MessageSender::errorSummaryMap) {
    // split the message category
    typedef boost::split_iterator<std::string::const_iterator> splitter;
    for (splitter i = boost::make_split_iterator(entry.first.category, boost::first_finder("|", boost::is_equal()));
         i != splitter();
         ++i)
    {
      // extract the substring corresponding to the split_iterator
      // FIXME: this can be avoided if the m_data map is keyed on boost::sub_range<std::string>
      category.assign(i->begin(), i->end());

      // access the message category, or create a new one as needed, and check the prescale
      if (getCategory(category).accept())
        accept = true;
    }
  }

  // harvest the errors, but only if the filter will accept the event
  std::auto_ptr<std::vector<edm::ErrorSummaryEntry> > errors(new std::vector<edm::ErrorSummaryEntry>());
  if (accept) {
    errors->reserve( edm::MessageSender::errorSummaryMap.size() );
    BOOST_FOREACH(const ErrorSummaryMap::value_type & entry, edm::MessageSender::errorSummaryMap) {
      errors->push_back(entry.first);        // sets category, module and severity
      errors->back().count = entry.second;   // count is 0 in key; set it to correct value (see FWCore/MessageLogger/src/LoggedErrorsSummary.cc)
    }
  }
  event.put(errors);

  // clear the errorSummaryMap
  edm::MessageSender::errorSummaryMap.clear();
  edm::MessageSender::freshError = false;

  return accept;
}

// ------------ method called at the end of the Job ---------
void HLTLogMonitorFilter::beginJob(void) {
  edm::MessageSender::errorSummaryIsBeingKept = true;
}
// ------------ method called at the end of the Job ---------
void HLTLogMonitorFilter::endJob(void) {
  edm::MessageSender::errorSummaryIsBeingKept = false;
  summary();
}

/// check if the requested category has a valid entry
bool HLTLogMonitorFilter::knownCategory(const std::string & category) {
  return (m_data.find( category ) != m_data.end());
}

/// create a new entry for the given category, with the given threshold value
HLTLogMonitorFilter::CategoryEntry & HLTLogMonitorFilter::addCategory(const std::string & category, uint32_t threshold) {
  // check after inserting, as either the new CategoryEntry is needed, or an error condition is raised
  std::pair<CategoryMap::iterator, bool> result = m_data.insert( std::make_pair(category, CategoryEntry(threshold)) );
  if (not result.second)
    throw cms::Exception("Configuration") << "Duplicate entry for category " << category;
  return result.first->second;
}

/// return the entry for requested category, if it exists, or create a new one with the default threshold value
HLTLogMonitorFilter::CategoryEntry & HLTLogMonitorFilter::getCategory(const std::string & category) {
  // check before inserting, to avoid the construction of a CategoryEntry object
  CategoryMap::iterator i = m_data.find(category);
  if (i != m_data.end())
    return i->second;
  else
    return m_data.insert( std::make_pair(category, CategoryEntry(m_prescale)) ).first->second;
}

/// summarize to LogInfo
void HLTLogMonitorFilter::summary(void) {
  std::stringstream out;
  out << "Log-Report ---------- HLTLogMonitorFilter Summary ------------\n"
      << "Log-Report  Threshold   Prescale     Issued   Accepted   Rejected Category\n";
  BOOST_FOREACH(const CategoryMap::value_type & entry, m_data) {
    out << "Log-Report "
        << std::right
        << std::setw(10) << entry.second.threshold << ' '
        << std::setw(10) << entry.second.prescale  << ' '
        << std::setw(10) << entry.second.counter   << ' '
        << std::setw(10) << entry.second.accepted  << ' '
        << std::setw(10) << (entry.second.counter - entry.second.accepted) << ' '
        << std::left     << entry.first            << '\n';
  }
  edm::LogVerbatim("Report") << out.str();
}

// define as a framework plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTLogMonitorFilter);
