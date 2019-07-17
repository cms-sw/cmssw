#include "DataFormats/Phase2L1ParticleFlow/interface/PFCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"

typedef Merger<std::vector<l1t::PFCandidate>> L1TPFCandMerger;

DEFINE_FWK_MODULE( L1TPFCandMerger );
