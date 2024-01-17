#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef Merger<reco::SuperClusterCollection> EgammaSuperClusterMerger;

template <>
void EgammaSuperClusterMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("src",
                                       {
                                           edm::InputTag("collection1"),
                                           edm::InputTag("collection2"),
                                       });
  descriptions.add("egammaSuperClusterMerger", desc);
}


DEFINE_FWK_MODULE(EgammaSuperClusterMerger);
