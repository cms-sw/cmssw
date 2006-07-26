#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/HLTexample/interface/HLTProdCand.h"
#include "HLTrigger/HLTexample/interface/HLTFiltCand.h"
#include "HLTrigger/HLTexample/interface/HLTAnalCand.h"
#include "HLTrigger/HLTexample/interface/HLTSimpleJet.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "HLTrigger/HLTexample/interface/HLTSinglet.h"
#include "HLTrigger/HLTexample/src/HLTSinglet.cc"

template HLTSinglet<reco::Electron> ;
template HLTSinglet<reco::Photon>   ;
template HLTSinglet<reco::Muon>     ;
template HLTSinglet<reco::CaloJet>  ;
template HLTSinglet<reco::CaloMET>  ;

typedef HLTSinglet<reco::Electron> HLT1Electron;
typedef HLTSinglet<reco::Photon>   HLT1Photon;
typedef HLTSinglet<reco::Muon>     HLT1Muon;
typedef HLTSinglet<reco::CaloJet>  HLT1CaloJet;
typedef HLTSinglet<reco::CaloMET>  HLT1CaloMET;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTProdCand)
DEFINE_ANOTHER_FWK_MODULE(HLTFiltCand)
DEFINE_ANOTHER_FWK_MODULE(HLTAnalCand)
DEFINE_ANOTHER_FWK_MODULE(HLTSimpleJet)
DEFINE_ANOTHER_FWK_MODULE(HLT1Electron);
DEFINE_ANOTHER_FWK_MODULE(HLT1Photon);
DEFINE_ANOTHER_FWK_MODULE(HLT1Muon);
DEFINE_ANOTHER_FWK_MODULE(HLT1CaloJet);
DEFINE_ANOTHER_FWK_MODULE(HLT1CaloMET);
