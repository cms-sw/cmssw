#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/HLTfilters/interface/HLTBool.h"
#include "HLTrigger/HLTfilters/interface/HLTFiltCand.h"
#include "HLTrigger/HLTfilters/interface/HLTLevel1GTSeed.h"
#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"
#include "HLTrigger/HLTfilters/interface/HLT1CaloJetEnergy.h"

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

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

using namespace reco;
using namespace trigger;

#include "HLTrigger/HLTfilters/interface/HLTSinglet.h"
#include "HLTrigger/HLTfilters/src/HLTSinglet.cc"

// template HLTSinglet<reco::Electron>             ;

typedef HLTSinglet<RecoEcalCandidate  ,TriggerPhoton> HLT1Photon   ;
typedef HLTSinglet<Electron         ,TriggerElectron> HLT1Electron ;
typedef HLTSinglet<RecoChargedCandidate ,TriggerMuon> HLT1Muon     ;
typedef HLTSinglet<CaloJet               ,TriggerTau> HLT1Tau      ;
typedef HLTSinglet<CaloJet               ,TriggerJet> HLT1CaloJet  ;
typedef HLTSinglet<CaloJet              ,TriggerBJet> HLT1CaloBJet ;
typedef HLTSinglet<CompositeCandidate             ,0> HLT1Composite;
typedef HLTSinglet<CaloMET               ,TriggerMET> HLT1CaloMET  ;
typedef HLTSinglet<MET                   ,TriggerMET> HLT1MET      ;
typedef HLTSinglet<RecoChargedCandidate,TriggerTrack> HLT1Track    ;
typedef HLTSinglet<RecoEcalCandidate, TriggerCluster> HLT1Cluster  ;

// filters for L1 candidates
typedef HLTSinglet<l1extra::L1EmParticle,     TriggerMuon> HLTLevel1Muon;
typedef HLTSinglet<l1extra::L1EtMissParticle,  TriggerMET> HLTLevel1MET;
typedef HLTSinglet<l1extra::L1HFRings,                  0> HLTLevel1HF;
typedef HLTSinglet<l1extra::L1JetParticle,     TriggerJet> HLTLevel1Jet;
typedef HLTSinglet<l1extra::L1MuonParticle,   TriggerMuon> HLTLevel1Muon;

#include "HLTrigger/HLTfilters/interface/HLTSmartSinglet.h"
#include "HLTrigger/HLTfilters/src/HLTSmartSinglet.cc"

// template HLTSmartSinglet<reco::Electron>             ;

typedef HLTSmartSinglet<RecoEcalCandidate  ,TriggerPhoton> HLT1SmartPhoton   ;
typedef HLTSmartSinglet<Electron         ,TriggerElectron> HLT1SmartElectron ;
typedef HLTSmartSinglet<RecoChargedCandidate ,TriggerMuon> HLT1SmartMuon     ;
typedef HLTSmartSinglet<CaloJet               ,TriggerTau> HLT1SmartTau      ;
typedef HLTSmartSinglet<CaloJet               ,TriggerJet> HLT1SmartCaloJet  ;
typedef HLTSmartSinglet<CaloJet              ,TriggerBJet> HLT1SmartCaloBJet ;
typedef HLTSmartSinglet<CompositeCandidate             ,0> HLT1SmartComposite;
typedef HLTSmartSinglet<CaloMET               ,TriggerMET> HLT1SmartCaloMET  ;
typedef HLTSmartSinglet<MET                   ,TriggerMET> HLT1SmartMET      ;
typedef HLTSmartSinglet<RecoChargedCandidate,TriggerTrack> HLT1SmartTrack    ;
typedef HLTSmartSinglet<RecoEcalCandidate ,TriggerCluster> HLT1SmartCluster  ;


#include "HLTrigger/HLTfilters/interface/HLTGlobalSums.h"
#include "HLTrigger/HLTfilters/src/HLTGlobalSums.cc"

//
typedef HLTGlobalSums<CaloMET,TriggerTET> HLTGlobalSumsCaloMET;
typedef HLTGlobalSums<MET    ,TriggerTHT> HLTGlobalSumsMET    ;

//

#include "HLTrigger/HLTfilters/interface/HLTDoublet.h"
#include "HLTrigger/HLTfilters/src/HLTDoublet.cc"
typedef HLTDoublet<CaloJet,TriggerJet,CaloJet,TriggerJet> HLT2JetJet;
typedef HLTDoublet<CaloJet,TriggerJet,CaloMET,TriggerMET> HLT2JetMET;
typedef HLTDoublet<Electron,TriggerElectron,CaloJet,TriggerTau> HLT2ElectronTau;
typedef HLTDoublet<RecoChargedCandidate,TriggerMuon,CaloJet,TriggerTau> HLT2MuonTau;

DEFINE_FWK_MODULE(HLTBool);
DEFINE_FWK_MODULE(HLTFiltCand);
DEFINE_FWK_MODULE(HLTLevel1GTSeed);
DEFINE_FWK_MODULE(HLTHighLevel);

DEFINE_FWK_MODULE(HLT2JetJet);
DEFINE_FWK_MODULE(HLT2JetMET);
DEFINE_FWK_MODULE(HLT2ElectronTau);
DEFINE_FWK_MODULE(HLT2MuonTau);

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
DEFINE_FWK_MODULE(HLT1CaloJetEnergy);

DEFINE_FWK_MODULE(HLTLevel1Muon);
DEFINE_FWK_MODULE(HLTLevel1MET);
DEFINE_FWK_MODULE(HLTLevel1HF);
DEFINE_FWK_MODULE(HLTLevel1Jet);
DEFINE_FWK_MODULE(HLTLevel1Muon);

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

DEFINE_FWK_MODULE(HLTGlobalSumsCaloMET);
DEFINE_FWK_MODULE(HLTGlobalSumsMET);
