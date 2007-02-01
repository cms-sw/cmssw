
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoJets/JetProducers/interface/KtJetProducer.h"
#include "RecoJets/JetProducers/interface/MidpointJetProducer.h"
#include "RecoJets/JetProducers/interface/IterativeConeJetProducer.h"
#include "RecoJets/JetProducers/interface/ToyJetCorrector.h"
#include "RecoJets/JetProducers/interface/FastJetProducer.h"
#include "RecoJets/JetProducers/interface/ExtKtJetProducer.h"


using cms::MidpointJetProducer;
using cms::KtJetProducer;
using cms::IterativeConeJetProducer;
using cms::ToyJetCorrector;
using cms::FastJetProducer;
using cms::ExtKtJetProducer;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MidpointJetProducer);
DEFINE_ANOTHER_FWK_MODULE(KtJetProducer);
DEFINE_ANOTHER_FWK_MODULE(IterativeConeJetProducer);
DEFINE_ANOTHER_FWK_MODULE(ToyJetCorrector);
DEFINE_ANOTHER_FWK_MODULE(FastJetProducer);
DEFINE_ANOTHER_FWK_MODULE(ExtKtJetProducer);
