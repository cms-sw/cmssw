#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "L3MumuTrackingRegion.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, L3MumuTrackingRegion, "L3MumuTrackingRegion");

#include "HLTJetTag.h"
DEFINE_FWK_MODULE(HLTJetTag);

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

#include "HLTCaloJetPairDzMatchFilter.h"
DEFINE_FWK_MODULE(HLTCaloJetPairDzMatchFilter);

#include "HLTCollectionProducer.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
typedef HLTCollectionProducer<reco::CaloJet> HLTCaloJetCollectionProducer;
typedef HLTCollectionProducer<reco::PFJet>   HLTPFJetCollectionProducer;
DEFINE_FWK_MODULE(HLTCaloJetCollectionProducer);
DEFINE_FWK_MODULE(HLTPFJetCollectionProducer);
