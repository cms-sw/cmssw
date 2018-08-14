#ifndef HLTFilter_h
#define HLTFilter_h

/** \class HLTFilter
 *
 *
 *  This class derives from EDFilter and adds a few HLT specific items.
 *  All HLT filters that wish to save summary objects for the AOD must derive from the HLTFilter class.
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include <string>
#include <utility>

//
// class decleration
//

class HLTFilter : public edm::global::EDFilter<> {

public:
  explicit HLTFilter(const edm::ParameterSet & config);
  static void makeHLTFilterDescription(edm::ParameterSetDescription& desc);
  ~HLTFilter() override;

private:
  bool filter(edm::StreamID, edm::Event & event, const edm::EventSetup & setup) const final;

  // declared pure virtual to enforce inheriting classes to implement it
  virtual bool hltFilter(edm::Event & event, const edm::EventSetup & setup, trigger::TriggerFilterObjectWithRefs & filterobject) const = 0;

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

#endif // HLTFilter_h
