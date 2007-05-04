
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoJets/JetProducers/interface/MidpointJetProducer.h"
#include "RecoJets/JetProducers/interface/IterativeConeJetProducer.h"
#include "RecoJets/JetProducers/interface/ToyJetCorrector.h"
#include "RecoJets/JetProducers/interface/FastJetProducer.h"
#include "RecoJets/JetProducers/interface/ExtKtJetProducer.h"

#include "RecoJets/JetProducers/interface/MidpointPilupSubtractionJetProducer.h"
#include "RecoJets/JetProducers/interface/IterativeConePilupSubtractionJetProducer.h"
#include "RecoJets/JetProducers/interface/FastPilupSubtractionJetProducer.h"
#include "RecoJets/JetProducers/interface/ExtKtPilupSubtractionJetProducer.h"


using cms::MidpointJetProducer;
using cms::IterativeConeJetProducer;
using cms::ToyJetCorrector;
using cms::FastJetProducer;
using cms::ExtKtJetProducer;

using cms::MidpointPilupSubtractionJetProducer;
using cms::IterativeConePilupSubtractionJetProducer;
using cms::FastPilupSubtractionJetProducer;
using cms::ExtKtPilupSubtractionJetProducer;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MidpointJetProducer);
DEFINE_ANOTHER_FWK_MODULE(IterativeConeJetProducer);
DEFINE_ANOTHER_FWK_MODULE(ToyJetCorrector);
DEFINE_ANOTHER_FWK_MODULE(FastJetProducer);
DEFINE_ANOTHER_FWK_MODULE(ExtKtJetProducer);
DEFINE_ANOTHER_FWK_MODULE(MidpointPilupSubtractionJetProducer);
DEFINE_ANOTHER_FWK_MODULE(IterativeConePilupSubtractionJetProducer);
DEFINE_ANOTHER_FWK_MODULE(FastPilupSubtractionJetProducer);
DEFINE_ANOTHER_FWK_MODULE(ExtKtPilupSubtractionJetProducer);
