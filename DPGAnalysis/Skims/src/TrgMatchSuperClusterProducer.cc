#include "DPGAnalysis/Skims/interface/TriggerMatchProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
typedef TriggerMatchProducer< reco::SuperCluster > trgMatchSuperClusterProducer;
DEFINE_FWK_MODULE( trgMatchSuperClusterProducer );
