
#include "PluginManager/ModuleDef.h"

#include "FWCore/CoreFramework/interface/MakerMacros.h"

#include "RecoJets/JetProducers/interface/KtJetProducer.h"
#include "RecoJets/JetProducers/interface/MidpointJetProducer.h"

namespace cms {
  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_MODULE(MidpointJetProducer)
  DEFINE_ANOTHER_FWK_MODULE(KtJetProducer)
}
