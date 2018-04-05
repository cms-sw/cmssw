#ifndef HLTStreamFilter_h
#define HLTStreamFilter_h

/** \class HLTStreamFilter
 *
 *
 *  This class derives from EDFilter and adds a few HLT specific items.
 *  All HLT filters that wish to save summary objects for the AOD must derive from the HLTStreamFilter class.
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include <string>
#include <utility>

//
// class decleration
//

class HLTStreamFilter : public edm::stream::EDFilter<> {

public:
  explicit HLTStreamFilter(const edm::ParameterSet & config);
  static void makeHLTFilterDescription(edm::ParameterSetDescription& desc);
  ~HLTStreamFilter() override;

private:
  bool filter(edm::Event & event, const edm::EventSetup & setup) final;

  // declared pure virtual to enforce inheriting classes to implement it
  virtual bool hltFilter(edm::Event & event, const edm::EventSetup & setup, trigger::TriggerFilterObjectWithRefs & filterobject) = 0;

private:
  const bool saveTags_;

public:
  bool saveTags() const {
    return saveTags_;
  }

public:
  int path(edm::Event const&) const;
  int module(edm::Event const&) const;
  std::pair<int,int> pmid(edm::Event const&) const;
  const std::string* pathName(edm::Event const&) const;
  const std::string* moduleLabel() const;
};

#endif // HLTStreamFilter_h
