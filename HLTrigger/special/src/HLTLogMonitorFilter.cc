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
// $Id: HLTLogMonitorFilter.cc,v 1.1 2009/11/09 00:01:23 fwyzard Exp $
//


// system include files
#include <stdint.h>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTLogMonitorFilter : public HLTFilter {
public:
    explicit HLTLogMonitorFilter(const edm::ParameterSet &);
    ~HLTLogMonitorFilter();

private:
    // ---------- private data types --------------------

    struct CategoryEntry {
      uint32_t threshold;       // configurable threshold, after which messages in this Category start to be logarithmically prescaled
      uint32_t prescale;        // current prescale for this Category
      uint64_t counter;         // number of events that fired this Category

      CategoryEntry(uint32_t t = 0) :
        threshold(t),           // default-constructed entries have the threshold set to 0, which means the associated category is disabled
        prescale(1),
        counter(0)
      { }
    };

    // ---------- private methods -----------------------

    /// EDFilter accept method
    virtual bool filter(edm::Event&, const edm::EventSetup &);

    /// check if the requested category has a valid entry
    bool knownCategory(const std::string & category);

    /// create a new entry for the given category, with the given threshold value
    CategoryEntry & addCategory(const std::string & category, uint32_t threshold);

    /// return the entry for requested category, if it exists, or create a new one with the default threshold value
    CategoryEntry & getCategory(const std::string & category);

    // ---------- member data ---------------------------
    uint32_t                            m_prescale;     // default threshold, after which messages in each Category start to be logarithmically prescaled
    std::map<std::string, CategoryEntry> m_data;        // map each category name to its prescale data
};


// system include files
#include <boost/foreach.hpp>
#include <boost/range.hpp>
#include <boost/algorithm/string.hpp>

// user include files
#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"

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
  BOOST_FOREACH(const edm::ParameterSet & category, categories) {
    const std::string & name = category.getParameter<std::string>("name");
    uint32_t threshold       = category.getParameter<uint32_t>("threshold");
    addCategory(name, threshold);
  }
}

HLTLogMonitorFilter::~HLTLogMonitorFilter()
{
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTLogMonitorFilter::filter(edm::Event & event, const edm::EventSetup & setup) {
  // no LogErrors or LogWarnings, skip processing and reject the event
  if (not edm::FreshErrorsExist())
    return false;

  // look at the LogErrors and LogWarnings, and accept the event if at least one is under threshold or matches the prescale
  bool accept = false;
  std::string category;
  std::vector<edm::ErrorSummaryEntry> errors = edm::LoggedErrorsSummary();  // returns by value and clears the internal log
  BOOST_FOREACH(const edm::ErrorSummaryEntry & error, errors) {
    // split the message category
    typedef boost::split_iterator<std::string::const_iterator> splitter;
    for (splitter i = boost::make_split_iterator(error.category, boost::first_finder("|", boost::is_equal()));
         i != splitter();
         ++i)
    {
      // extract the substring corresponding to the split_iterator
      // FIXME: this can be avoided if the m_data map is keyed on boost::sub_range<std::string>
      category.assign(i->begin(), i->end());

      // access the message category, or create a new one as needed
      CategoryEntry & data = getCategory(category);
      ++data.counter;

      // if this category is disabled (scale == 0), skip this event
      // if the counter is not a multiple of the prescale, skip the event
      if ((data.prescale == 0) or (data.counter % data.prescale))
        continue;

      // quasi-logarithmic increase in the prescale factor (should be safe even if threshold is 1)
      if (data.counter == data.prescale * data.threshold)
        data.prescale *= data.threshold;

      accept = true;
    }
  }

  return accept;
}

/// check if the requested category has a valid entry
bool HLTLogMonitorFilter::knownCategory(const std::string & category) {
  return (m_data.find( category ) != m_data.end());
}

/// create a new entry for the given category, with the given threshold value
HLTLogMonitorFilter::CategoryEntry & HLTLogMonitorFilter::addCategory(const std::string & category, uint32_t threshold) {
  // check after inserting, as either the new CategoryEntry is needed, or an error condition is raised
  std::pair<std::map<std::string, HLTLogMonitorFilter::CategoryEntry>::iterator, bool> result = m_data.insert( std::make_pair(category, CategoryEntry(threshold)) );
  if (result.second)
    throw cms::Exception("Configuration") << "Duplicate entry for category " << category;
  return result.first->second;
}

/// return the entry for requested category, if it exists, or create a new one with the default threshold value
HLTLogMonitorFilter::CategoryEntry & HLTLogMonitorFilter::getCategory(const std::string & category) {
  // check before inserting, to avoid the construction of a CategoryEntry object
  std::map<std::string, HLTLogMonitorFilter::CategoryEntry>::iterator i = m_data.find(category);
  if (i != m_data.end())
    return i->second;
  else
    return m_data.insert( std::make_pair(category, CategoryEntry(m_prescale)) ).first->second;
}


// define as a framework plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTLogMonitorFilter);
