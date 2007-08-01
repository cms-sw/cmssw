#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/HLTfilters/interface/HLTFiltCand.h"
#include "HLTrigger/HLTfilters/interface/HLTLevel1Seed.h"
#include "HLTrigger/HLTfilters/interface/HLTLevel1GTSeed.h"
#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"
#include "HLTrigger/HLTfilters/interface/HLTDoublet.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"


#include "HLTrigger/HLTfilters/interface/HLTSinglet.h"
#include "HLTrigger/HLTfilters/src/HLTSinglet.cc"

// template HLTSinglet<reco::Electron>             ;

typedef HLTSinglet<reco::Electron> HLT1Electron ;
typedef HLTSinglet<reco::Photon>   HLT1Photon   ;
typedef HLTSinglet<reco::Muon>     HLT1Muon     ;
typedef HLTSinglet<reco::CaloJet>  HLT1Tau      ; // taus are stored as jets
typedef HLTSinglet<reco::CaloJet>  HLT1CaloJet  ;
typedef HLTSinglet<reco::CaloMET>  HLT1CaloMET  ;
typedef HLTSinglet<reco::MET>      HLT1CaloHT   ;
typedef HLTSinglet<reco::RecoChargedCandidate> HLT1Track;
typedef HLTSinglet<reco::RecoEcalCandidate>    HLT1SuperCluster;


#include "HLTrigger/HLTfilters/interface/HLTSmartSinglet.h"
#include "HLTrigger/HLTfilters/src/HLTSmartSinglet.cc"

// template HLTSmartSinglet<reco::Electron>             ;

typedef HLTSmartSinglet<reco::Electron> HLT1SmartElectron ;
typedef HLTSmartSinglet<reco::Photon>   HLT1SmartPhoton   ;
typedef HLTSmartSinglet<reco::Muon>     HLT1SmartMuon     ;
typedef HLTSmartSinglet<reco::CaloJet>  HLT1SmartTau      ; // taus are stored as jets
typedef HLTSmartSinglet<reco::CaloJet>  HLT1SmartCaloJet  ;
typedef HLTSmartSinglet<reco::CaloMET>  HLT1SmartCaloMET  ;
typedef HLTSmartSinglet<reco::MET>      HLT1SmartCaloHT   ;
typedef HLTSmartSinglet<reco::RecoChargedCandidate> HLT1SmartTrack;
typedef HLTSmartSinglet<reco::RecoEcalCandidate>    HLT1SmartSuperCluster;


#include "HLTrigger/HLTfilters/interface/HLTGlobalSums.h"
#include "HLTrigger/HLTfilters/src/HLTGlobalSums.cc"

//

typedef HLTGlobalSums<reco::CaloMET>  HLTGlobalSumMET  ;
typedef HLTGlobalSums<reco::MET>      HLTGlobalSumHT   ;


DEFINE_FWK_MODULE(HLTFiltCand);
DEFINE_FWK_MODULE(HLTLevel1Seed);
DEFINE_FWK_MODULE(HLTLevel1GTSeed);
DEFINE_FWK_MODULE(HLTHighLevel);
DEFINE_FWK_MODULE(HLTDoublet);

DEFINE_FWK_MODULE(HLT1Electron);
DEFINE_FWK_MODULE(HLT1Photon);
DEFINE_FWK_MODULE(HLT1Muon);
DEFINE_FWK_MODULE(HLT1Tau);
DEFINE_FWK_MODULE(HLT1CaloJet);
DEFINE_FWK_MODULE(HLT1CaloMET);
DEFINE_FWK_MODULE(HLT1CaloHT);
DEFINE_FWK_MODULE(HLT1Track);
DEFINE_FWK_MODULE(HLT1SuperCluster);

DEFINE_FWK_MODULE(HLT1SmartElectron);
DEFINE_FWK_MODULE(HLT1SmartPhoton);
DEFINE_FWK_MODULE(HLT1SmartMuon);
DEFINE_FWK_MODULE(HLT1SmartTau);
DEFINE_FWK_MODULE(HLT1SmartCaloJet);
DEFINE_FWK_MODULE(HLT1SmartCaloMET);
DEFINE_FWK_MODULE(HLT1SmartCaloHT);
DEFINE_FWK_MODULE(HLT1SmartTrack);
DEFINE_FWK_MODULE(HLT1SmartSuperCluster);

DEFINE_FWK_MODULE(HLTGlobalSumMET);
DEFINE_FWK_MODULE(HLTGlobalSumHT);
