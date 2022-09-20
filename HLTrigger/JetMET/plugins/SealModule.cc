#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//No changes
#include "HLTrigger/JetMET/interface/AnyJetToCaloJetProducer.h"
#include "HLTrigger/JetMET/interface/HLT2jetGapFilter.h"
#include "HLTrigger/JetMET/interface/HLTAcoFilter.h"
#include "HLTrigger/JetMET/interface/HLTHemiDPhiFilter.h"
#include "HLTrigger/JetMET/interface/HLTPhi2METFilter.h"
#include "HLTrigger/JetMET/interface/HLTRapGapFilter.h"
#include "HLTrigger/JetMET/interface/HLTPFEnergyFractionsFilter.h"
#include "HLTrigger/JetMET/interface/HLTHtMhtFilter.h"
#include "HLTrigger/JetMET/interface/HLTMhtFilter.h"
#include "HLTrigger/JetMET/interface/HLTHPDFilter.h"
#include "HLTrigger/JetMET/interface/HLTHcalMETNoiseCleaner.h"
#include "HLTrigger/JetMET/interface/HLTHcalMETNoiseFilter.h"
#include "HLTrigger/JetMET/interface/HLTHcalLaserFilter.h"
#include "HLTrigger/JetMET/interface/HLTHcalTowerNoiseCleaner.h"
#include "HLTrigger/JetMET/interface/HLTHcalTowerNoiseCleanerWithrechit.h"
#include "HLTrigger/JetMET/interface/HLTNVFilter.h"
#include "HLTrigger/JetMET/interface/HLTCaloJetIDProducer.h"
#include "HLTrigger/JetMET/interface/HLTPFJetIDProducer.h"
#include "HLTrigger/JetMET/interface/HLTMETCleanerUsingJetID.h"

#include "HLTrigger/JetMET/plugins/HLTJetHFCleaner.h"

//Work with all jet collections without changing the module name
#include "HLTrigger/JetMET/interface/HLTHtMhtProducer.h"
#include "HLTrigger/JetMET/interface/HLTCaloTowerHtMhtProducer.h"
#include "HLTrigger/JetMET/interface/HLTMhtProducer.h"
#include "HLTrigger/JetMET/interface/HLTTrackMETProducer.h"
#include "HLTrigger/JetMET/interface/HLTMinDPhiMETFilter.h"

//Template
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
//
#include "HLTrigger/JetMET/interface/HLTAlphaTFilter.h"
#include "HLTrigger/JetMET/src/HLTAlphaTFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTDiJetAveFilter.h"
#include "HLTrigger/JetMET/src/HLTDiJetAveFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTDiJetAveEtaFilter.h"
#include "HLTrigger/JetMET/src/HLTDiJetAveEtaFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTDiJetEtaTopologyFilter.h"
#include "HLTrigger/JetMET/src/HLTDiJetEtaTopologyFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTJetEtaTopologyFilter.h"
#include "HLTrigger/JetMET/src/HLTJetEtaTopologyFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTJetSortedVBFFilter.h"
#include "HLTrigger/JetMET/src/HLTJetSortedVBFFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTJetL1MatchProducer.h"
#include "HLTrigger/JetMET/src/HLTJetL1MatchProducer.cc"
//
#include "HLTrigger/JetMET/interface/HLTJetL1TMatchProducer.h"
#include "HLTrigger/JetMET/src/HLTJetL1TMatchProducer.cc"
//
#include "HLTrigger/JetMET/interface/HLTMonoJetFilter.h"
#include "HLTrigger/JetMET/src/HLTMonoJetFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTJetCollForElePlusJets.h"
#include "HLTrigger/JetMET/src/HLTJetCollForElePlusJets.cc"
//
#include "HLTrigger/JetMET/interface/HLTJetCollectionsForElePlusJets.h"
#include "HLTrigger/JetMET/src/HLTJetCollectionsForElePlusJets.cc"
//
#include "HLTrigger/JetMET/interface/HLTJetCollectionsForLeptonPlusJets.h"
#include "HLTrigger/JetMET/src/HLTJetCollectionsForLeptonPlusJets.cc"
//
//
#include "HLTrigger/JetMET/interface/HLTJetCollectionsForBoostedLeptonPlusJets.h"
#include "HLTrigger/JetMET/src/HLTJetCollectionsForBoostedLeptonPlusJets.cc"
//
#include "HLTrigger/JetMET/interface/HLTJetsCleanedFromLeadingLeptons.h"
//
#include "HLTrigger/JetMET/interface/HLTJetCollectionsFilter.h"
#include "HLTrigger/JetMET/src/HLTJetCollectionsFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTJetCollectionsVBFFilter.h"
#include "HLTrigger/JetMET/src/HLTJetCollectionsVBFFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTJetVBFFilter.h"
#include "HLTrigger/JetMET/src/HLTJetVBFFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTForwardBackwardJetsFilter.h"
#include "HLTrigger/JetMET/src/HLTForwardBackwardJetsFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTFatJetMassFilter.h"
#include "HLTrigger/JetMET/src/HLTFatJetMassFilter.cc"
//
#include "HLTrigger/JetMET/interface/HLTExclDiJetFilter.h"
#include "HLTrigger/JetMET/src/HLTExclDiJetFilter.cc"
//
#include "HLTJetsMatchedToFilteredJetsProducer.h"

using namespace reco;
using namespace trigger;

typedef HLTAlphaTFilter<CaloJet> HLTAlphaTCaloJetFilter;
typedef HLTAlphaTFilter<PFJet> HLTAlphaTPFJetFilter;

typedef HLTDiJetAveFilter<CaloJet> HLTDiCaloJetAveFilter;
typedef HLTDiJetAveFilter<PFJet> HLTDiPFJetAveFilter;

typedef HLTDiJetAveEtaFilter<CaloJet> HLTDiCaloJetAveEtaFilter;
typedef HLTDiJetAveEtaFilter<PFJet> HLTDiPFJetAveEtaFilter;

typedef HLTDiJetEtaTopologyFilter<CaloJet> HLTDiCaloJetEtaTopologyFilter;
typedef HLTDiJetEtaTopologyFilter<PFJet> HLTDiPFJetEtaTopologyFilter;

typedef HLTJetEtaTopologyFilter<CaloJet> HLTCaloJetEtaTopologyFilter;
typedef HLTJetEtaTopologyFilter<PFJet> HLTPFJetEtaTopologyFilter;

typedef HLTJetSortedVBFFilter<CaloJet> HLTCaloJetSortedVBFFilter;
typedef HLTJetSortedVBFFilter<PFJet> HLTPFJetSortedVBFFilter;

typedef HLTJetL1MatchProducer<CaloJet> HLTCaloJetL1MatchProducer;
typedef HLTJetL1MatchProducer<PFJet> HLTPFJetL1MatchProducer;

typedef HLTJetL1TMatchProducer<CaloJet> HLTCaloJetL1TMatchProducer;
typedef HLTJetL1TMatchProducer<PFJet> HLTPFJetL1TMatchProducer;

typedef HLTMonoJetFilter<CaloJet> HLTMonoCaloJetFilter;
typedef HLTMonoJetFilter<PFJet> HLTMonoPFJetFilter;

typedef HLTJetCollForElePlusJets<CaloJet> HLTCaloJetCollForElePlusJets;
typedef HLTJetCollForElePlusJets<PFJet> HLTPFJetCollForElePlusJets;

typedef HLTJetCollectionsForElePlusJets<CaloJet> HLTCaloJetCollectionsForElePlusJets;
typedef HLTJetCollectionsForElePlusJets<PFJet> HLTPFJetCollectionsForElePlusJets;

typedef HLTJetCollectionsForBoostedLeptonPlusJets<PFJet> HLTPFJetCollectionsForBoostedLeptonPlusJets;

typedef HLTJetCollectionsForLeptonPlusJets<CaloJet> HLTCaloJetCollectionsForLeptonPlusJets;
typedef HLTJetCollectionsForLeptonPlusJets<PFJet> HLTPFJetCollectionsForLeptonPlusJets;

typedef HLTJetsCleanedFromLeadingLeptons<CaloJet> HLTCaloJetsCleanedFromLeadingLeptons;
typedef HLTJetsCleanedFromLeadingLeptons<PFJet> HLTPFJetsCleanedFromLeadingLeptons;

typedef HLTJetCollectionsFilter<CaloJet> HLTCaloJetCollectionsFilter;
typedef HLTJetCollectionsFilter<PFJet> HLTPFJetCollectionsFilter;

typedef HLTJetCollectionsVBFFilter<CaloJet> HLTCaloJetCollectionsVBFFilter;
typedef HLTJetCollectionsVBFFilter<PFJet> HLTPFJetCollectionsVBFFilter;

typedef HLTJetVBFFilter<CaloJet> HLTCaloJetVBFFilter;
typedef HLTJetVBFFilter<PFJet> HLTPFJetVBFFilter;

typedef HLTForwardBackwardJetsFilter<CaloJet> HLTForwardBackwardCaloJetsFilter;
typedef HLTForwardBackwardJetsFilter<PFJet> HLTForwardBackwardPFJetsFilter;

typedef HLTFatJetMassFilter<CaloJet> HLTFatCaloJetMassFilter;
typedef HLTFatJetMassFilter<PFJet> HLTFatPFJetMassFilter;

typedef HLTExclDiJetFilter<CaloJet> HLTExclDiCaloJetFilter;
typedef HLTExclDiJetFilter<PFJet> HLTExclDiPFJetFilter;

//No changes
DEFINE_FWK_MODULE(AnyJetToCaloJetProducer);
DEFINE_FWK_MODULE(HLT2jetGapFilter);
DEFINE_FWK_MODULE(HLTAcoFilter);
DEFINE_FWK_MODULE(HLTHemiDPhiFilter);
DEFINE_FWK_MODULE(HLTPhi2METFilter);
DEFINE_FWK_MODULE(HLTRapGapFilter);
DEFINE_FWK_MODULE(HLTPFEnergyFractionsFilter);
DEFINE_FWK_MODULE(HLTHtMhtFilter);
DEFINE_FWK_MODULE(HLTMhtFilter);
DEFINE_FWK_MODULE(HLTHPDFilter);
DEFINE_FWK_MODULE(HLTCaloJetIDProducer);
DEFINE_FWK_MODULE(HLTPFJetIDProducer);
DEFINE_FWK_MODULE(HLTHcalMETNoiseFilter);
DEFINE_FWK_MODULE(HLTHcalMETNoiseCleaner);
DEFINE_FWK_MODULE(HLTHcalLaserFilter);
DEFINE_FWK_MODULE(HLTHcalTowerNoiseCleaner);
DEFINE_FWK_MODULE(HLTHcalTowerNoiseCleanerWithrechit);
DEFINE_FWK_MODULE(HLTNVFilter);
DEFINE_FWK_MODULE(HLTMETCleanerUsingJetID);

//Work with all jet collections without changing the module name
DEFINE_FWK_MODULE(HLTMhtProducer);
DEFINE_FWK_MODULE(HLTHtMhtProducer);
DEFINE_FWK_MODULE(HLTCaloTowerHtMhtProducer);
DEFINE_FWK_MODULE(HLTTrackMETProducer);
DEFINE_FWK_MODULE(HLTMinDPhiMETFilter);

//Templates

DEFINE_FWK_MODULE(HLTAlphaTCaloJetFilter);
DEFINE_FWK_MODULE(HLTAlphaTPFJetFilter);

DEFINE_FWK_MODULE(HLTCaloJetSortedVBFFilter);
DEFINE_FWK_MODULE(HLTPFJetSortedVBFFilter);

DEFINE_FWK_MODULE(HLTMonoCaloJetFilter);
DEFINE_FWK_MODULE(HLTMonoPFJetFilter);

DEFINE_FWK_MODULE(HLTCaloJetCollectionsFilter);
DEFINE_FWK_MODULE(HLTPFJetCollectionsFilter);

DEFINE_FWK_MODULE(HLTCaloJetCollectionsVBFFilter);
DEFINE_FWK_MODULE(HLTPFJetCollectionsVBFFilter);

DEFINE_FWK_MODULE(HLTCaloJetCollForElePlusJets);
DEFINE_FWK_MODULE(HLTPFJetCollForElePlusJets);

DEFINE_FWK_MODULE(HLTCaloJetCollectionsForElePlusJets);
DEFINE_FWK_MODULE(HLTPFJetCollectionsForElePlusJets);

DEFINE_FWK_MODULE(HLTPFJetCollectionsForBoostedLeptonPlusJets);

DEFINE_FWK_MODULE(HLTCaloJetCollectionsForLeptonPlusJets);
DEFINE_FWK_MODULE(HLTPFJetCollectionsForLeptonPlusJets);

DEFINE_FWK_MODULE(HLTCaloJetsCleanedFromLeadingLeptons);
DEFINE_FWK_MODULE(HLTPFJetsCleanedFromLeadingLeptons);

DEFINE_FWK_MODULE(HLTDiCaloJetAveFilter);
DEFINE_FWK_MODULE(HLTDiPFJetAveFilter);

DEFINE_FWK_MODULE(HLTDiCaloJetAveEtaFilter);
DEFINE_FWK_MODULE(HLTDiPFJetAveEtaFilter);

DEFINE_FWK_MODULE(HLTDiCaloJetEtaTopologyFilter);
DEFINE_FWK_MODULE(HLTDiPFJetEtaTopologyFilter);

DEFINE_FWK_MODULE(HLTCaloJetEtaTopologyFilter);
DEFINE_FWK_MODULE(HLTPFJetEtaTopologyFilter);

DEFINE_FWK_MODULE(HLTCaloJetL1MatchProducer);
DEFINE_FWK_MODULE(HLTPFJetL1MatchProducer);

DEFINE_FWK_MODULE(HLTCaloJetL1TMatchProducer);
DEFINE_FWK_MODULE(HLTPFJetL1TMatchProducer);

DEFINE_FWK_MODULE(HLTCaloJetVBFFilter);
DEFINE_FWK_MODULE(HLTPFJetVBFFilter);

DEFINE_FWK_MODULE(HLTForwardBackwardCaloJetsFilter);
DEFINE_FWK_MODULE(HLTForwardBackwardPFJetsFilter);

DEFINE_FWK_MODULE(HLTFatCaloJetMassFilter);
DEFINE_FWK_MODULE(HLTFatPFJetMassFilter);

DEFINE_FWK_MODULE(HLTExclDiCaloJetFilter);
DEFINE_FWK_MODULE(HLTExclDiPFJetFilter);

typedef HLTJetsMatchedToFilteredJetsProducer<reco::CaloJet, reco::CaloJetRef>
    HLTCaloJetsMatchedToFilteredCaloJetsProducer;
DEFINE_FWK_MODULE(HLTCaloJetsMatchedToFilteredCaloJetsProducer);

typedef HLTJetsMatchedToFilteredJetsProducer<reco::CaloJet, reco::PFJetRef> HLTCaloJetsMatchedToFilteredPFJetsProducer;
DEFINE_FWK_MODULE(HLTCaloJetsMatchedToFilteredPFJetsProducer);

typedef HLTJetsMatchedToFilteredJetsProducer<reco::PFJet, reco::CaloJetRef> HLTPFJetsMatchedToFilteredCaloJetsProducer;
DEFINE_FWK_MODULE(HLTPFJetsMatchedToFilteredCaloJetsProducer);

typedef HLTJetsMatchedToFilteredJetsProducer<reco::PFJet, reco::PFJetRef> HLTPFJetsMatchedToFilteredPFJetsProducer;
DEFINE_FWK_MODULE(HLTPFJetsMatchedToFilteredPFJetsProducer);

typedef HLTJetHFCleaner<reco::CaloJet> HLTCaloJetHFCleaner;
DEFINE_FWK_MODULE(HLTCaloJetHFCleaner);

typedef HLTJetHFCleaner<reco::PFJet> HLTPFJetHFCleaner;
DEFINE_FWK_MODULE(HLTPFJetHFCleaner);
