
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoJets/JetProducers/interface/KtJetProducer.h"
#include "RecoJets/JetProducers/interface/MidpointJetProducer.h"
#include "RecoJets/JetProducers/interface/MidpointJetProducer2.h"
#include "RecoJets/JetProducers/interface/ToyJetCorrector.h"

using cms::MidpointJetProducer;
using cms::MidpointJetProducer2;
using cms::KtJetProducer;
using cms::ToyJetCorrector;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MidpointJetProducer)
DEFINE_ANOTHER_FWK_MODULE(MidpointJetProducer2)
DEFINE_ANOTHER_FWK_MODULE(KtJetProducer)
DEFINE_ANOTHER_FWK_MODULE(ToyJetCorrector)
