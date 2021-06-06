#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTBool.h"
#include "HLTFiltCand.h"
#include "HLTHighLevel.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/TauReco/interface/HLTTau.h"
#include "DataFormats/TauReco/interface/HLTTauFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

using namespace reco;
using namespace trigger;

#include "HLTSinglet.h"
#include "HLTSinglet.cc"

#include "L1TTkEleFilter.h"
#include "L1TTkEmFilter.h"
#include "L1TTkMuonFilter.h"
#include "L1TPFTauFilter.h"
#include "L1THPSPFTauFilter.h"
#include "L1TJetFilterT.h"
#include "L1TEnergySumFilterT.h"

// filter for HLT candidates
typedef HLTSinglet<RecoEcalCandidate> HLT1Photon;
typedef HLTSinglet<Electron> HLT1Electron;
typedef HLTSinglet<RecoChargedCandidate> HLT1Muon;
typedef HLTSinglet<CaloJet> HLT1CaloJet;
typedef HLTSinglet<CompositeCandidate> HLT1Composite;
typedef HLTSinglet<CaloMET> HLT1CaloMET;
typedef HLTSinglet<MET> HLT1MET;
//typedef HLTSinglet<RecoChargedCandidate> HLT1Track    ;
//typedef HLTSinglet<RecoEcalCandidate   > HLT1Cluster  ;
typedef HLTSinglet<PFTau> HLT1PFTau;
typedef HLTSinglet<PFJet> HLT1PFJet;
typedef HLTSinglet<PFMET> HLT1PFMET;

// filters for L1 candidates
typedef HLTSinglet<l1extra::L1EmParticle>
    HLTLevel1EG;  // the actual type is ovrridden object-by-object (TriggerL1IsoEG or TriggerL1NoIsoEG)
typedef HLTSinglet<l1extra::L1EtMissParticle>
    HLTLevel1MET;  // the actual type is ovrridden object-by-object (TriggerL1ETM or TriggerL1HTM)
typedef HLTSinglet<l1extra::L1JetParticle>
    HLTLevel1Jet;  // the actual type is ovrridden object-by-object (TriggerL1CenJet, TriggerL1ForJet or TriggerL1TauJet)
typedef HLTSinglet<l1extra::L1MuonParticle> HLTLevel1Muon;

// filters for Phase-2
typedef L1TJetFilterT<reco::CaloJet> L1TJetFilter;
typedef L1TJetFilterT<l1t::PFJet> L1TPFJetFilter;
typedef L1TEnergySumFilterT<reco::MET> L1TEnergySumFilter;
typedef L1TEnergySumFilterT<reco::PFMET> L1TPFEnergySumFilter;

#include "HLTSmartSinglet.h"
#include "HLTSmartSinglet.cc"

typedef HLTSmartSinglet<RecoEcalCandidate> HLT1SmartPhoton;
typedef HLTSmartSinglet<Electron> HLT1SmartElectron;
typedef HLTSmartSinglet<RecoChargedCandidate> HLT1SmartMuon;
typedef HLTSmartSinglet<CaloJet> HLT1SmartCaloJet;
typedef HLTSmartSinglet<CompositeCandidate> HLT1SmartComposite;
typedef HLTSmartSinglet<CaloMET> HLT1SmartCaloMET;
typedef HLTSmartSinglet<MET> HLT1SmartMET;
typedef HLTSmartSinglet<PFTau> HLT1SmartPFTau;
typedef HLTSmartSinglet<PFJet> HLT1SmartPFJet;
typedef HLTSmartSinglet<PFMET> HLT1SmartPFMET;

#include "HLTGlobalSums.h"
#include "HLTGlobalSums.cc"

typedef HLTGlobalSums<PFMET> HLTGlobalSumsPFMET;
typedef HLTGlobalSums<CaloMET> HLTGlobalSumsCaloMET;
typedef HLTGlobalSums<MET> HLTGlobalSumsMET;

#include "HLTDoublet.h"
#include "HLTDoublet.cc"
typedef HLTDoublet<CaloJet, CaloJet> HLT2CaloJetCaloJet;
typedef HLTDoublet<CaloJet, CaloMET> HLT2CaloJetCaloMET;
typedef HLTDoublet<CaloJet, MET> HLT2CaloJetMET;
typedef HLTDoublet<PFJet, PFJet> HLT2PFJetPFJet;
typedef HLTDoublet<PFJet, CaloMET> HLT2PFJetCaloMET;
typedef HLTDoublet<PFJet, MET> HLT2PFJetMET;
typedef HLTDoublet<PFJet, PFMET> HLT2PFJetPFMET;

typedef HLTDoublet<Electron, CaloJet> HLT2ElectronTau;
typedef HLTDoublet<RecoEcalCandidate, CaloJet> HLT2PhotonTau;
typedef HLTDoublet<RecoChargedCandidate, CaloJet> HLT2MuonTau;
typedef HLTDoublet<Electron, HLTTau> HLT2ElectronHLTTau;
typedef HLTDoublet<RecoChargedCandidate, HLTTau> HLT2MuonHLTTau;
typedef HLTDoublet<Electron, PFTau> HLT2ElectronPFTau;
typedef HLTDoublet<RecoEcalCandidate, PFTau> HLT2PhotonPFTau;
typedef HLTDoublet<RecoChargedCandidate, PFTau> HLT2MuonPFTau;
typedef HLTDoublet<Electron, PFJet> HLT2ElectronPFJet;
typedef HLTDoublet<RecoChargedCandidate, PFJet> HLT2MuonPFJet;

typedef HLTDoublet<Electron, CaloMET> HLT2ElectronCaloMET;
typedef HLTDoublet<RecoChargedCandidate, CaloMET> HLT2MuonCaloMET;
typedef HLTDoublet<Electron, MET> HLT2ElectronMET;
typedef HLTDoublet<RecoChargedCandidate, MET> HLT2MuonMET;
typedef HLTDoublet<Electron, PFMET> HLT2ElectronPFMET;
typedef HLTDoublet<RecoChargedCandidate, PFMET> HLT2MuonPFMET;
typedef HLTDoublet<RecoEcalCandidate, MET> HLT2PhotonMET;
typedef HLTDoublet<RecoEcalCandidate, PFMET> HLT2PhotonPFMET;

DEFINE_FWK_MODULE(HLTBool);
DEFINE_FWK_MODULE(HLTFiltCand);
DEFINE_FWK_MODULE(HLTHighLevel);

DEFINE_FWK_MODULE(HLT2CaloJetCaloJet);
DEFINE_FWK_MODULE(HLT2CaloJetCaloMET);
DEFINE_FWK_MODULE(HLT2CaloJetMET);
DEFINE_FWK_MODULE(HLT2PFJetPFJet);
DEFINE_FWK_MODULE(HLT2PFJetCaloMET);
DEFINE_FWK_MODULE(HLT2PFJetMET);
DEFINE_FWK_MODULE(HLT2PFJetPFMET);
DEFINE_FWK_MODULE(HLT2ElectronTau);
DEFINE_FWK_MODULE(HLT2PhotonTau);
DEFINE_FWK_MODULE(HLT2MuonTau);
DEFINE_FWK_MODULE(HLT2ElectronPFTau);
DEFINE_FWK_MODULE(HLT2PhotonPFTau);
DEFINE_FWK_MODULE(HLT2MuonPFTau);
DEFINE_FWK_MODULE(HLT2ElectronPFJet);
DEFINE_FWK_MODULE(HLT2MuonPFJet);
DEFINE_FWK_MODULE(HLT2ElectronCaloMET);
DEFINE_FWK_MODULE(HLT2MuonCaloMET);
DEFINE_FWK_MODULE(HLT2ElectronMET);
DEFINE_FWK_MODULE(HLT2MuonMET);
DEFINE_FWK_MODULE(HLT2ElectronPFMET);
DEFINE_FWK_MODULE(HLT2MuonPFMET);
DEFINE_FWK_MODULE(HLT2PhotonMET);
DEFINE_FWK_MODULE(HLT2PhotonPFMET);

DEFINE_FWK_MODULE(HLT1Electron);
DEFINE_FWK_MODULE(HLT1Photon);
DEFINE_FWK_MODULE(HLT1Muon);
DEFINE_FWK_MODULE(HLT1CaloJet);
DEFINE_FWK_MODULE(HLT1Composite);
DEFINE_FWK_MODULE(HLT1CaloMET);
DEFINE_FWK_MODULE(HLT1MET);
DEFINE_FWK_MODULE(HLT1PFTau);
DEFINE_FWK_MODULE(HLT1PFJet);
DEFINE_FWK_MODULE(HLT1PFMET);

DEFINE_FWK_MODULE(HLTLevel1EG);
DEFINE_FWK_MODULE(HLTLevel1MET);
DEFINE_FWK_MODULE(HLTLevel1Jet);
DEFINE_FWK_MODULE(HLTLevel1Muon);

// Phase-2
DEFINE_FWK_MODULE(L1TTkEleFilter);
DEFINE_FWK_MODULE(L1TTkEmFilter);
DEFINE_FWK_MODULE(L1TTkMuonFilter);
DEFINE_FWK_MODULE(L1TPFTauFilter);
DEFINE_FWK_MODULE(L1THPSPFTauFilter);
DEFINE_FWK_MODULE(L1TJetFilter);
DEFINE_FWK_MODULE(L1TPFJetFilter);
DEFINE_FWK_MODULE(L1TEnergySumFilter);
DEFINE_FWK_MODULE(L1TPFEnergySumFilter);

DEFINE_FWK_MODULE(HLTGlobalSumsPFMET);
DEFINE_FWK_MODULE(HLTGlobalSumsCaloMET);
DEFINE_FWK_MODULE(HLTGlobalSumsMET);
