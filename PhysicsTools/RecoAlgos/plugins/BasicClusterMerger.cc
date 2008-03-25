#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

typedef Merger<reco::BasicClusterCollection> BasicClusterMerger;

DEFINE_FWK_MODULE( BasicClusterMerger );
