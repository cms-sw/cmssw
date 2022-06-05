#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef Merger<reco::SuperClusterCollection> EgammaSuperClusterMerger;
DEFINE_FWK_MODULE(EgammaSuperClusterMerger);
