
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/JetReco/interface/BasicJet.h"

#include "KtJetProducer.h"
#include "MidpointJetProducer.h"
#include "CDFMidpointJetProducer.h"
#include "SISConeJetProducer.h"
#include "IterativeConeJetProducer.h"
#include "ExtKtJetProducer.h"

#include "MidpointPilupSubtractionJetProducer.h"
#include "IterativeConePilupSubtractionJetProducer.h"
#include "KtPilupSubtractionJetProducer.h"
#include "ExtKtPilupSubtractionJetProducer.h"

#include "PtMinJetSelector.h"

#include "InputGenJetsParticleSelector.h"


using cms::KtJetProducer;
using cms::MidpointJetProducer;
using cms::CDFMidpointJetProducer;
using cms::SISConeJetProducer;
using cms::IterativeConeJetProducer;
using cms::ExtKtJetProducer;

using cms::MidpointPilupSubtractionJetProducer;
using cms::IterativeConePilupSubtractionJetProducer;
using cms::KtPilupSubtractionJetProducer;
using cms::ExtKtPilupSubtractionJetProducer;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(KtJetProducer);
DEFINE_ANOTHER_FWK_MODULE(IterativeConeJetProducer);
DEFINE_ANOTHER_FWK_MODULE(MidpointJetProducer);
DEFINE_ANOTHER_FWK_MODULE(CDFMidpointJetProducer);
DEFINE_ANOTHER_FWK_MODULE(SISConeJetProducer);
DEFINE_ANOTHER_FWK_MODULE(ExtKtJetProducer);

DEFINE_ANOTHER_FWK_MODULE(MidpointPilupSubtractionJetProducer);
DEFINE_ANOTHER_FWK_MODULE(IterativeConePilupSubtractionJetProducer);
DEFINE_ANOTHER_FWK_MODULE(KtPilupSubtractionJetProducer);
DEFINE_ANOTHER_FWK_MODULE(ExtKtPilupSubtractionJetProducer);

DEFINE_ANOTHER_FWK_MODULE(PtMinCaloJetSelector);
DEFINE_ANOTHER_FWK_MODULE(PtMinGenJetSelector);
DEFINE_ANOTHER_FWK_MODULE(PtMinPFJetSelector);
DEFINE_ANOTHER_FWK_MODULE(PtMinBasicJetSelector);

DEFINE_ANOTHER_FWK_MODULE(InputGenJetsParticleSelector);
