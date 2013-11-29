#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "L3MumuTrackingRegion.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, L3MumuTrackingRegion, "L3MumuTrackingRegion");


#include "HLTDisplacedmumuFilter.h"
DEFINE_FWK_MODULE(HLTDisplacedmumuFilter);

#include "HLTDisplacedmumuVtxProducer.h"
DEFINE_FWK_MODULE(HLTDisplacedmumuVtxProducer);

#include "HLTDisplacedmumumuFilter.h"
DEFINE_FWK_MODULE(HLTDisplacedmumumuFilter);

#include "HLTDisplacedmumumuVtxProducer.h"
DEFINE_FWK_MODULE(HLTDisplacedmumumuVtxProducer);

#include "HLTmmkFilter.h"
DEFINE_FWK_MODULE(HLTmmkFilter);

#include "HLTmmkkFilter.h"
DEFINE_FWK_MODULE(HLTmmkkFilter);

#include "ConeIsolation.h"
DEFINE_FWK_MODULE(ConeIsolation);

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include "HLTJetPairDzMatchFilter.h"
#include "HLTJetPairDzMatchFilter.cc"
typedef   HLTJetPairDzMatchFilter<reco::CaloJet> HLTCaloJetPairDzMatchFilter;
typedef   HLTJetPairDzMatchFilter<reco::PFJet> HLTPFJetPairDzMatchFilter;
typedef   HLTJetPairDzMatchFilter<reco::PFTau> HLTPFTauPairDzMatchFilter;
DEFINE_FWK_MODULE(HLTCaloJetPairDzMatchFilter);
DEFINE_FWK_MODULE(HLTPFJetPairDzMatchFilter);
DEFINE_FWK_MODULE(HLTPFTauPairDzMatchFilter);

#include "HLTJetTag.h"
#include "HLTJetTag.cc"
typedef HLTJetTag<reco::CaloJet> HLTCaloJetTag;
typedef HLTJetTag<reco::  PFJet> HLTPFJetTag;
DEFINE_FWK_MODULE(HLTCaloJetTag);
DEFINE_FWK_MODULE(HLTPFJetTag);

#include "HLTCollectionProducer.h"
typedef HLTCollectionProducer<reco::CaloJet> HLTCaloJetCollectionProducer;
typedef HLTCollectionProducer<reco::PFJet>   HLTPFJetCollectionProducer;
DEFINE_FWK_MODULE(HLTCaloJetCollectionProducer);
DEFINE_FWK_MODULE(HLTPFJetCollectionProducer);
