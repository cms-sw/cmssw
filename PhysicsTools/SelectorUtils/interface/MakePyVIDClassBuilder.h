#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

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
  
};
