#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "MCJetCorrectionService.h"
#include "JetCorrectionProducer.h"
using namespace cms;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(JetCorrectionProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(MCJetCorrectionService);
