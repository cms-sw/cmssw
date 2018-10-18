#ifndef __PhysicsTools_SelectorUtils_MakePyVIDClassBuilder_h__
#define __PhysicsTools_SelectorUtils_MakePyVIDClassBuilder_h__

#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"

template<class PhysObj>
struct MakeVersionedSelector {
  MakeVersionedSelector() {}
  
  VersionedSelector<edm::Ptr<PhysObj> > 
  operator()(const std::string& pset, 
             const std::string& which_config) {
    const edm::ParameterSet& temp = 
      edm::readPSetsFrom(pset)->getParameter<edm::ParameterSet>(which_config);
    return VersionedSelector<edm::Ptr<PhysObj> >(temp);
  }

  VersionedSelector<edm::Ptr<PhysObj> > 
  operator()() {
    return VersionedSelector<edm::Ptr<PhysObj> >();
  }
  
};

#endif
