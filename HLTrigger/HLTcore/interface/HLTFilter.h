#ifndef HLTFilter_h
#define HLTFilter_h

/** \class HLTFilter
 *
 *  
 *  This class derives from EDFilter and adds a few HLT specific items.
 *  All HLT filters that wish to save summary objects for the AOD must derive from the HLTFilter class.
 *
 *  $Date: 2012/02/01 13:50:55 $
 *  $Revision: 1.10 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

//
// class decleration
//

class HLTFilter : public edm::EDFilter {

public:
  explicit HLTFilter(const edm::ParameterSet & config);
  static void makeHLTFilterDescription(edm::ParameterSetDescription& desc);
  virtual ~HLTFilter();

private:
  bool filter(edm::Event & event, const edm::EventSetup & setup);

  // declared pue virtual to enforce inheriting classes to implement it
  virtual bool hltFilter(edm::Event & event, const edm::EventSetup & setup, trigger::TriggerFilterObjectWithRefs & filterobject) = 0;

private:
  const bool saveTags_;

public:
  bool saveTags() const {
    return saveTags_;
  }

public:
  int path() const;
  int module() const;
  std::pair<int,int> pmid() const;
  const std::string* pathName() const;
  const std::string* moduleLabel() const;
};

#endif // HLTFilter_h
