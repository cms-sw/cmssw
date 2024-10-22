#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"

typedef SingleObjectSelector<std::vector<l1t::PFCandidate>, StringCutObjectSelector<l1t::PFCandidate>> L1TPFCandSelector;

DEFINE_FWK_MODULE(L1TPFCandSelector);
