
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoJets/JetProducers/interface/MidpointPilupSubtractionJetProducer.h"
#include "RecoJets/JetProducers/interface/IterativeConePilupSubtractionJetProducer.h"
#include "RecoJets/JetProducers/interface/FastPilupSubtractionJetProducer.h"
#include "RecoJets/JetProducers/interface/ExtKtPilupSubtractionJetProducer.h"


using cms::MidpointPilupSubtractionJetProducer;
using cms::IterativeConePilupSubtractionJetProducer;
using cms::FastPilupSubtractionJetProducer;
using cms::ExtKtPilupSubtractionJetProducer;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MidpointPilupSubtractionJetProducer);
DEFINE_ANOTHER_FWK_MODULE(IterativeConePilupSubtractionJetProducer);
DEFINE_ANOTHER_FWK_MODULE(FastPilupSubtractionJetProducer);
DEFINE_ANOTHER_FWK_MODULE(ExtKtPilupSubtractionJetProducer);
