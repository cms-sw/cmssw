#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_SEAL_MODULE();

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/Modules/interface/JetCorrectionService.h"
#include "JetMETCorrections/Modules/interface/JetCorrectionServiceChain.h"
#include "JetMETCorrections/Modules/interface/JetCorrectionProducer.h"
#include "JetMETCorrections/Modules/interface/PlotJetCorrections.h"
#include "JetMETCorrections/Algorithms/interface/ZSPJetCorrector.h"
#include "JetMETCorrections/Algorithms/interface/LXXXCorrector.h"
#include "JetMETCorrections/Algorithms/interface/JetPlusTrackCorrector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
REGISTER_PLUGIN (JetCorrectionsRecord, JetCorrector);

using namespace cms;
using namespace reco;

typedef JetCorrectionProducer<CaloJet> CaloJetCorrectionProducer;
DEFINE_ANOTHER_FWK_MODULE(CaloJetCorrectionProducer);

typedef JetCorrectionProducer<PFJet> PFJetCorrectionProducer;
DEFINE_ANOTHER_FWK_MODULE(PFJetCorrectionProducer);

DEFINE_ANOTHER_FWK_MODULE(PlotJetCorrections);

DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(JetCorrectionServiceChain);

//--------------- Generic LX correction service --------------------
DEFINE_JET_CORRECTION_SERVICE (LXXXCorrector, LXXXCorrectionService);

//--------------- Zero suppression correction service --------------
DEFINE_JET_CORRECTION_SERVICE (ZSPJetCorrector, ZSPJetCorrectionService);

//--------------- JPT correction service ---------------------------
DEFINE_JET_CORRECTION_SERVICE (JetPlusTrackCorrector, JetPlusTrackCorrectionService);

