#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "EgammaHLTFilteredObjProducer.h"

template <>
void EgammaHLTFilteredObjProducer<std::vector<reco::SuperClusterRef>>::addObj(
    const reco::RecoEcalCandidateRef& cand, std::vector<reco::SuperClusterRef>& output) {
  output.push_back(cand->superCluster());
}

using EgammaHLTFilteredSuperClusterProducer = EgammaHLTFilteredObjProducer<std::vector<reco::SuperClusterRef>>;
DEFINE_FWK_MODULE(EgammaHLTFilteredSuperClusterProducer);
