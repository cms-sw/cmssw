#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/JetMET/interface/HLT2jetGapFilter.h"
#include "HLTrigger/JetMET/interface/HLTAcoFilter.h"
#include "HLTrigger/JetMET/interface/HLTDiJetAveFilter.h"
#include "HLTrigger/JetMET/interface/HLTExclDiJetFilter.h"
#include "HLTrigger/JetMET/interface/HLTForwardBackwardJetsFilter.h"
#include "HLTrigger/JetMET/interface/HLTHcalMETNoiseFilter.h"
#include "HLTrigger/JetMET/interface/HLTHcalMETNoiseCleaner.h"
#include "HLTrigger/JetMET/interface/HLTHcalTowerNoiseCleaner.h"
#include "HLTrigger/JetMET/interface/HLTHPDFilter.h"
#include "HLTrigger/JetMET/interface/HLTJetVBFFilter.h"
#include "HLTrigger/JetMET/interface/HLTJetCollectionsVBFFilter.h"
#include "HLTrigger/JetMET/interface/HLTMhtHtFilter.h"
#include "HLTrigger/JetMET/interface/HLTNVFilter.h"
#include "HLTrigger/JetMET/interface/HLTPhi2METFilter.h"
#include "HLTrigger/JetMET/interface/HLTRapGapFilter.h"
#include "HLTrigger/JetMET/interface/HLTJetIDProducer.h"
#include "HLTrigger/JetMET/interface/HLTJetCollForElePlusJets.h"
#include "HLTrigger/JetMET/interface/HLTMhtFilter.h"
#include "HLTrigger/JetMET/interface/HLTMhtProducer.h"
#include "HLTrigger/JetMET/interface/HLTHtMhtProducer.h"
#include "HLTrigger/JetMET/interface/HLTHtMhtFilter.h"

#include "HLTrigger/JetMET/interface/HLTJetSortedVBFFilter.h"
#include "HLTrigger/JetMET/src/HLTJetSortedVBFFilter.cc"

#include "HLTrigger/JetMET/interface/HLTJetL1MatchProducer.h"
#include "HLTrigger/JetMET/src/HLTJetL1MatchProducer.cc"

#include "HLTrigger/JetMET/interface/HLTPFEnergyFractionsFilter.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "HLTrigger/JetMET/interface/HLTMonoJetFilter.h"
#include "HLTrigger/JetMET/src/HLTMonoJetFilter.cc"
#include "HLTrigger/JetMET/interface/HLTJetCollectionsForLeptonPlusJets.h"
#include "HLTrigger/JetMET/src/HLTJetCollectionsForLeptonPlusJets.cc"
#include "HLTrigger/JetMET/interface/HLTJetCollectionsFilter.h"
#include "HLTrigger/JetMET/src/HLTJetCollectionsFilter.cc"
#include "HLTrigger/JetMET/interface/HLTAlphaTFilter.h"

#include "HLTrigger/JetMET/interface/PFJetsMatchedToFilteredCaloJetsProducer.h"

using namespace reco;
using namespace trigger;

typedef HLTMonoJetFilter<CaloJet> HLTMonoCaloJetFilter;
typedef HLTMonoJetFilter<  PFJet> HLTMonoPFJetFilter;

typedef HLTJetL1MatchProducer<CaloJet> HLTCaloJetL1MatchProducer;
typedef HLTJetL1MatchProducer<PFJet> HLTPFJetL1MatchProducer;

typedef HLTJetCollectionsFilter<CaloJet> HLTCaloJetCollectionsFilter;
typedef HLTJetCollectionsFilter<  PFJet> HLTPFJetCollectionsFilter;

typedef HLTJetCollectionsForLeptonPlusJets<CaloJet> HLTCaloJetCollectionsForLeptonPlusJets;
typedef HLTJetCollectionsForLeptonPlusJets<  PFJet> HLTPFJetCollectionsForLeptonPlusJets;

typedef HLTJetSortedVBFFilter<CaloJet> HLTCaloJetSortedVBFFilter;
typedef HLTJetSortedVBFFilter<  PFJet> HLTPFJetSortedVBFFilter;

DEFINE_FWK_MODULE(HLT2jetGapFilter);
DEFINE_FWK_MODULE(HLTAcoFilter);
DEFINE_FWK_MODULE(HLTDiJetAveFilter);
DEFINE_FWK_MODULE(HLTExclDiJetFilter);
DEFINE_FWK_MODULE(HLTForwardBackwardJetsFilter);
DEFINE_FWK_MODULE(HLTHcalMETNoiseFilter);
DEFINE_FWK_MODULE(HLTHcalMETNoiseCleaner);
DEFINE_FWK_MODULE(HLTHcalTowerNoiseCleaner);
DEFINE_FWK_MODULE(HLTHPDFilter);
DEFINE_FWK_MODULE(HLTJetVBFFilter);
DEFINE_FWK_MODULE(HLTJetCollectionsVBFFilter);
DEFINE_FWK_MODULE(HLTMhtHtFilter);
DEFINE_FWK_MODULE(HLTNVFilter);
DEFINE_FWK_MODULE(HLTPhi2METFilter);
DEFINE_FWK_MODULE(HLTRapGapFilter);
DEFINE_FWK_MODULE(HLTJetIDProducer);

DEFINE_FWK_MODULE(HLTCaloJetL1MatchProducer);
DEFINE_FWK_MODULE(HLTPFJetL1MatchProducer);

DEFINE_FWK_MODULE(HLTJetCollForElePlusJets);
DEFINE_FWK_MODULE(HLTMhtFilter);
DEFINE_FWK_MODULE(HLTMhtProducer);
DEFINE_FWK_MODULE(HLTHtMhtProducer);
DEFINE_FWK_MODULE(HLTHtMhtFilter);

DEFINE_FWK_MODULE(HLTCaloJetSortedVBFFilter);
DEFINE_FWK_MODULE(HLTPFJetSortedVBFFilter);

DEFINE_FWK_MODULE(HLTPFEnergyFractionsFilter);

DEFINE_FWK_MODULE(HLTMonoCaloJetFilter);
DEFINE_FWK_MODULE(HLTMonoPFJetFilter);

DEFINE_FWK_MODULE(HLTCaloJetCollectionsFilter);
DEFINE_FWK_MODULE(HLTPFJetCollectionsFilter);

DEFINE_FWK_MODULE(HLTCaloJetCollectionsForLeptonPlusJets);
DEFINE_FWK_MODULE(HLTPFJetCollectionsForLeptonPlusJets);

DEFINE_FWK_MODULE(HLTAlphaTFilter);

DEFINE_FWK_MODULE(PFJetsMatchedToFilteredCaloJetsProducer);
