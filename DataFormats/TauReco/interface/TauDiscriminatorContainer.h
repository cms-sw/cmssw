#ifndef DataFormats_TauReco_TauDiscriminatorContainer_h
#define DataFormats_TauReco_TauDiscriminatorContainer_h
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/RefProd.h"

#include <vector>

namespace reco {
  struct SingleTauDiscriminatorContainer {
    std::vector<float> rawValues; // stores floating point discriminators, like MVA raw values or pt sums.
    std::vector<bool> workingPoints; // stores boolean discriminators computed with the raw values.

    SingleTauDiscriminatorContainer() {}
    SingleTauDiscriminatorContainer(float rawInit) { rawValues.push_back(rawInit); }
  };

  typedef edm::ValueMap<SingleTauDiscriminatorContainer> TauDiscriminatorContainer;

  //typedef pat::PATTauDiscriminatorContainer::value_type PATTauDiscriminatorContainerVT;
  //typedef edm::Ref<pat::PATTauDiscriminatorContainer> PATTauDiscriminatorContainerRef;
  //typedef edm::RefProd<pat::PATTauDiscriminatorContainer> PATTauDiscriminatorContainerRefProd;
  //typedef edm::RefVector<pat::PATTauDiscriminatorContainer> PATTauDiscriminatorContainerRefVector;
}  // namespace reco

#endif
