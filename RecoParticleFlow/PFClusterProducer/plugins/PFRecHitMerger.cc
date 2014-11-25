#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

typedef Merger<std::vector<reco::PFRecHit> > PFRecHitMerger;


DEFINE_FWK_MODULE( PFRecHitMerger );
