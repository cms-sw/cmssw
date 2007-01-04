#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "MCJetCorrectionService.h"
#include "JetCorrectionProducer.h"
using namespace cms;
DEFINE_SEAL_MODULE();
REGISTER_PLUGIN (JetCorrectionsRecord, JetCorrector);
DEFINE_ANOTHER_FWK_MODULE(JetCorrectionProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(MCJetCorrectionService);
