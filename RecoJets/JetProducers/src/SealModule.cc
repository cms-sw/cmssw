
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoJets/JetProducers/interface/KtJetProducer.h"
#include "RecoJets/JetProducers/interface/MidpointJetProducer.h"
#include "RecoJets/JetProducers/interface/IterativeConeJetProducer.h"
#include "RecoJets/JetProducers/interface/ToyJetCorrector.h"

using cms::MidpointJetProducer;
using cms::KtJetProducer;
using cms::IterativeConeJetProducer;
using cms::ToyJetCorrector;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MidpointJetProducer)
DEFINE_ANOTHER_FWK_MODULE(KtJetProducer)
DEFINE_ANOTHER_FWK_MODULE(IterativeConeJetProducer)
DEFINE_ANOTHER_FWK_MODULE(ToyJetCorrector)
