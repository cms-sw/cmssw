#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

typedef Merger<reco::SuperClusterCollection> SuperClusterMerger;

DEFINE_FWK_MODULE( SuperClusterMerger );
