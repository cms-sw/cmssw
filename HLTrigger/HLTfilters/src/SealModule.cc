#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/HLTfilters/interface/HLTBool.h"
#include "HLTrigger/HLTfilters/interface/HLTFiltCand.h"
#include "HLTrigger/HLTfilters/interface/HLTLevel1GTSeed.h"
#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"

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
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"
#include "DataFormats/TauReco/interface/HLTTau.h"
#include "DataFormats/TauReco/interface/HLTTauFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include <typeinfo>

using namespace reco;
using namespace trigger;

#include "HLTrigger/HLTfilters/interface/HLTSinglet.h"
#include "HLTrigger/HLTfilters/src/HLTSinglet.cc"

// filter for HLT candidates
typedef HLTSinglet<RecoEcalCandidate   > HLT1Photon   ;
typedef HLTSinglet<Electron            > HLT1Electron ;
typedef HLTSinglet<RecoChargedCandidate> HLT1Muon     ;
typedef HLTSinglet<CaloJet             > HLT1Tau      ; // obsolete - use HLT1CaloJet
typedef HLTSinglet<CaloJet             > HLT1CaloJet  ;
typedef HLTSinglet<CaloJet             > HLT1CaloBJet ; // obsolete - use HLT1CaloJet 
typedef HLTSinglet<CompositeCandidate  > HLT1Composite;
typedef HLTSinglet<CaloMET             > HLT1CaloMET  ;
typedef HLTSinglet<MET                 > HLT1MET      ;
typedef HLTSinglet<RecoChargedCandidate> HLT1Track    ;
typedef HLTSinglet<RecoEcalCandidate   > HLT1Cluster  ;
typedef HLTSinglet<PFTau               > HLT1PFTau    ;
typedef HLTSinglet<PFJet               > HLT1PFJet    ;
typedef HLTSinglet<PFJet               > HLT1PFBJet   ; // obsolete - use HLT1PFJet 

// filters for L1 candidates
typedef HLTSinglet<l1extra::L1EmParticle    > HLTLevel1EG;     // the actual type is ovrridden object-by-object (TriggerL1IsoEG or TriggerL1NoIsoEG)
typedef HLTSinglet<l1extra::L1EtMissParticle> HLTLevel1MET;    // the actual type is ovrridden object-by-object (TriggerL1ETM or TriggerL1HTM)
typedef HLTSinglet<l1extra::L1JetParticle   > HLTLevel1Jet;    // the actual type is ovrridden object-by-object (TriggerL1CenJet, TriggerL1ForJet or TriggerL1TauJet)
typedef HLTSinglet<l1extra::L1MuonParticle  > HLTLevel1Muon;   

#include "HLTrigger/HLTfilters/interface/HLTSmartSinglet.h"
#include "HLTrigger/HLTfilters/src/HLTSmartSinglet.cc"

typedef HLTSmartSinglet<RecoEcalCandidate   > HLT1SmartPhoton   ;
typedef HLTSmartSinglet<Electron            > HLT1SmartElectron ;
typedef HLTSmartSinglet<RecoChargedCandidate> HLT1SmartMuon     ;
typedef HLTSmartSinglet<CaloJet             > HLT1SmartTau      ; // obsoleted by HLT1SmartCaloJet
typedef HLTSmartSinglet<CaloJet             > HLT1SmartCaloJet  ;
typedef HLTSmartSinglet<CaloJet             > HLT1SmartCaloBJet ; // obsoleted by HLT1SmartCaloJet
typedef HLTSmartSinglet<CompositeCandidate  > HLT1SmartComposite;
typedef HLTSmartSinglet<CaloMET             > HLT1SmartCaloMET  ;
typedef HLTSmartSinglet<MET                 > HLT1SmartMET      ;
typedef HLTSmartSinglet<RecoChargedCandidate> HLT1SmartTrack    ;
typedef HLTSmartSinglet<RecoEcalCandidate   > HLT1SmartCluster  ;
typedef HLTSmartSinglet<PFTau               > HLT1SmartPFTau    ;
typedef HLTSmartSinglet<PFJet               > HLT1SmartPFJet    ;
typedef HLTSmartSinglet<PFJet               > HLT1SmartPFBJet   ; // obsoleted by HLT1SmartCaloJet


#include "HLTrigger/HLTfilters/interface/HLTGlobalSums.h"
#include "HLTrigger/HLTfilters/src/HLTGlobalSums.cc"

typedef HLTGlobalSums<CaloMET> HLTGlobalSumsCaloMET;
typedef HLTGlobalSums<MET    > HLTGlobalSumsMET    ;


#include "HLTrigger/HLTfilters/interface/HLTDoublet.h"
#include "HLTrigger/HLTfilters/src/HLTDoublet.cc"
typedef HLTDoublet<CaloJet,CaloJet> HLT2CaloJetCaloJet;
typedef HLTDoublet<CaloJet,CaloMET> HLT2CaloJetCaloMET;
typedef HLTDoublet<CaloJet,    MET> HLT2CaloJetMETMHT; // obsoleted by HLT2CaloJetMET
typedef HLTDoublet<CaloJet,    MET> HLT2CaloJetMETTHT; // obsoleted by HLT2CaloJetMET
typedef HLTDoublet<CaloJet,    MET> HLT2CaloJetMET;

typedef HLTDoublet<Electron            ,CaloJet> HLT2ElectronTau;
typedef HLTDoublet<RecoChargedCandidate,CaloJet> HLT2MuonTau;
typedef HLTDoublet<Electron            ,CaloTau> HLT2ElectronCaloTau;
typedef HLTDoublet<RecoChargedCandidate,CaloTau> HLT2MuonCaloTau;
typedef HLTDoublet<Electron            ,HLTTau>  HLT2ElectronHLTTau;
typedef HLTDoublet<RecoChargedCandidate,HLTTau>  HLT2MuonHLTTau;
typedef HLTDoublet<Electron            ,PFTau>   HLT2ElectronPFTau;
typedef HLTDoublet<RecoChargedCandidate,PFTau>   HLT2MuonPFTau;

typedef HLTDoublet<Electron            ,CaloMET> HLT2ElectronCaloMET;
typedef HLTDoublet<RecoChargedCandidate,CaloMET> HLT2MuonCaloMET;
typedef HLTDoublet<Electron            ,    MET> HLT2ElectronMET;
typedef HLTDoublet<RecoChargedCandidate,    MET> HLT2MuonMET;

#include "HLTrigger/HLTfilters/interface/HLTDoubletDZ.h"
#include "HLTrigger/HLTfilters/src/HLTDoubletDZ.cc"
typedef HLTDoubletDZ<Electron            ,Electron>             HLT2ElectronElectronDZ;
typedef HLTDoubletDZ<RecoChargedCandidate,RecoChargedCandidate> HLT2MuonMuonDZ;
typedef HLTDoubletDZ<Electron            ,RecoChargedCandidate> HLT2ElectronMuonDZ;

DEFINE_FWK_MODULE(HLTBool);
DEFINE_FWK_MODULE(HLTFiltCand);
DEFINE_FWK_MODULE(HLTLevel1GTSeed);
DEFINE_FWK_MODULE(HLTHighLevel);

DEFINE_FWK_MODULE(HLT2CaloJetCaloJet);
DEFINE_FWK_MODULE(HLT2CaloJetCaloMET);
DEFINE_FWK_MODULE(HLT2CaloJetMETMHT);
DEFINE_FWK_MODULE(HLT2CaloJetMETTHT);
DEFINE_FWK_MODULE(HLT2CaloJetMET);
DEFINE_FWK_MODULE(HLT2ElectronTau);
DEFINE_FWK_MODULE(HLT2MuonTau);
//DEFINE_FWK_MODULE(HLT2ElectronCaloTau);
//DEFINE_FWK_MODULE(HLT2MuonCaloTau);
//DEFINE_FWK_MODULE(HLT2ElectronHLTTau);
//DEFINE_FWK_MODULE(HLT2MuonHLTTau);
DEFINE_FWK_MODULE(HLT2ElectronPFTau);
DEFINE_FWK_MODULE(HLT2MuonPFTau);
DEFINE_FWK_MODULE(HLT2ElectronCaloMET);
DEFINE_FWK_MODULE(HLT2MuonCaloMET);
DEFINE_FWK_MODULE(HLT2ElectronMET);
DEFINE_FWK_MODULE(HLT2MuonMET);


DEFINE_FWK_MODULE(HLT1Electron);
DEFINE_FWK_MODULE(HLT1Photon);
DEFINE_FWK_MODULE(HLT1Muon);
DEFINE_FWK_MODULE(HLT1Tau);
DEFINE_FWK_MODULE(HLT1CaloJet);
DEFINE_FWK_MODULE(HLT1CaloBJet);
DEFINE_FWK_MODULE(HLT1Composite);
DEFINE_FWK_MODULE(HLT1CaloMET);
DEFINE_FWK_MODULE(HLT1MET);
DEFINE_FWK_MODULE(HLT1Track);
DEFINE_FWK_MODULE(HLT1Cluster);
DEFINE_FWK_MODULE(HLT1PFTau);
DEFINE_FWK_MODULE(HLT1PFJet);
DEFINE_FWK_MODULE(HLT1PFBJet);

DEFINE_FWK_MODULE(HLTLevel1EG);
DEFINE_FWK_MODULE(HLTLevel1MET);
DEFINE_FWK_MODULE(HLTLevel1Jet);
DEFINE_FWK_MODULE(HLTLevel1Muon);

/*
DEFINE_FWK_MODULE(HLT1SmartElectron);
DEFINE_FWK_MODULE(HLT1SmartPhoton);
DEFINE_FWK_MODULE(HLT1SmartMuon);
DEFINE_FWK_MODULE(HLT1SmartTau);
DEFINE_FWK_MODULE(HLT1SmartCaloJet);
DEFINE_FWK_MODULE(HLT1SmartCaloBJet);
DEFINE_FWK_MODULE(HLT1SmartComposite);
DEFINE_FWK_MODULE(HLT1SmartCaloMET);
DEFINE_FWK_MODULE(HLT1SmartMET);
DEFINE_FWK_MODULE(HLT1SmartTrack);
DEFINE_FWK_MODULE(HLT1SmartCluster);
DEFINE_FWK_MODULE(HLT1SmartPFTau);
DEFINE_FWK_MODULE(HLT1SmartPFJet);
DEFINE_FWK_MODULE(HLT1SmartPFBJet);
*/

DEFINE_FWK_MODULE(HLTGlobalSumsCaloMET);
DEFINE_FWK_MODULE(HLTGlobalSumsMET);

DEFINE_FWK_MODULE(HLT2ElectronElectronDZ);
DEFINE_FWK_MODULE(HLT2MuonMuonDZ);
DEFINE_FWK_MODULE(HLT2ElectronMuonDZ);
