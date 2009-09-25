#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

using namespace cms;

#include "JetMETCorrections/Modules/interface/JetCorrectionProducer.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
typedef JetCorrectionProducer<pat::Jet> PatJetCorrectionProducer;
DEFINE_ANOTHER_FWK_MODULE(PatJetCorrectionProducer);

#include "FWCore/Framework/interface/SourceFactory.h"
#include "JetMETCorrections/Modules/interface/JetCorrectionService.h"
#include "JetMETCorrections/JetPlusTrack/interface/PatJPTCorrector.h"
DEFINE_JET_CORRECTION_SERVICE(PatJPTCorrector,PatJPTCorrectionService);
