#include "CondCore/PluginSystem/interface/registration_macros.h"
DEFINE_SEAL_MODULE();

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
REGISTER_PLUGIN (JetCorrectionsRecord, JetCorrector);

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

using namespace cms;

#include "JetCorrectionProducer.h"
DEFINE_ANOTHER_FWK_MODULE(JetCorrectionProducer);

#include "JetCorrectionService.icc"
#include "JetMETCorrections/MCJet/interface/MCJetCorrector.h"
DEFINE_JET_CORRECTION_SERVICE (MCJetCorrector, MCJetCorrectionService);
