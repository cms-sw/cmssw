#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTauTag/HLTProducers/interface/PFTauToJetProducer.h"
#include "RecoTauTag/HLTProducers/interface/PFJetToCaloProducer.h"
#include "RecoTauTag/HLTProducers/interface/L1HLTJetsMatching.h"
#include "RecoTauTag/HLTProducers/interface/L1HLTTauMatching.h"
#include "RecoTauTag/HLTProducers/interface/L1THLTTauMatching.h"
#include "RecoTauTag/HLTProducers/interface/L2TauJetsMerger.h"
#include "RecoTauTag/HLTProducers/interface/CaloTowerCreatorForTauHLT.h"
#include "RecoTauTag/HLTProducers/interface/CaloTowerFromL1TCreatorForTauHLT.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h" 	 
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h" 	 
#include "TauRegionalPixelSeedGenerator.h" 	 
#include "TrackingRegionsFromBeamSpotAndL2Tau.h"
#include "CandidateSeededTrackingRegionsProducer.h"
#include "RecoTauTag/HLTProducers/interface/TauJetSelectorForHLTTrackSeeding.h"
#include "RecoTauTag/HLTProducers/interface/VertexFromTrackProducer.h"
//#include "RecoTauTag/HLTProducers/interface/L2TauPixelTrackMatch.h"
#include "HLTPFTauPairLeadTrackDzMatchFilter.h"
#include "RecoTauTag/HLTProducers/interface/L2TauPixelIsoTagProducer.h"
#include "RecoTauTag/HLTProducers/interface/PFJetsTauOverlapRemoval.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "RecoTauTag/HLTProducers/interface/L1TJetsMatching.h"

typedef L1TJetsMatching<reco::PFJet> L1TPFJetsMatching ;
typedef L1TJetsMatching<reco::CaloJet> L1TCaloJetsMatching ;

DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, TauRegionalPixelSeedGenerator, "TauRegionalPixelSeedGenerator");      
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, TrackingRegionsFromBeamSpotAndL2Tau, "TrackingRegionsFromBeamSpotAndL2Tau");
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, CandidateSeededTrackingRegionsProducer, "CandidateSeededTrackingRegionsProducer");

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionEDProducerT.h"
using TauRegionalPixelSeedTrackingRegionEDProducer = TrackingRegionEDProducerT<TauRegionalPixelSeedGenerator>;
DEFINE_FWK_MODULE(TauRegionalPixelSeedTrackingRegionEDProducer);
using CandidateSeededTrackingRegionsEDProducer = TrackingRegionEDProducerT<CandidateSeededTrackingRegionsProducer>;
DEFINE_FWK_MODULE(CandidateSeededTrackingRegionsEDProducer);
using TrackingRegionsFromBeamSpotAndL2TauEDProducer = TrackingRegionEDProducerT<TrackingRegionsFromBeamSpotAndL2Tau>;
DEFINE_FWK_MODULE(TrackingRegionsFromBeamSpotAndL2TauEDProducer);

DEFINE_FWK_MODULE(L2TauJetsMerger);
DEFINE_FWK_MODULE(L1HLTJetsMatching);
DEFINE_FWK_MODULE(L1HLTTauMatching);
DEFINE_FWK_MODULE(L1THLTTauMatching);
DEFINE_FWK_MODULE(CaloTowerCreatorForTauHLT);
DEFINE_FWK_MODULE(CaloTowerFromL1TCreatorForTauHLT);
DEFINE_FWK_MODULE(PFTauToJetProducer);
DEFINE_FWK_MODULE(PFJetToCaloProducer);
DEFINE_FWK_MODULE(TauJetSelectorForHLTTrackSeeding);
DEFINE_FWK_MODULE(VertexFromTrackProducer);
//DEFINE_FWK_MODULE(L2TauPixelTrackMatch);
DEFINE_FWK_MODULE(HLTPFTauPairLeadTrackDzMatchFilter);
DEFINE_FWK_MODULE(L2TauPixelIsoTagProducer);
DEFINE_FWK_MODULE(L1TCaloJetsMatching);
DEFINE_FWK_MODULE(L1TPFJetsMatching);
DEFINE_FWK_MODULE(PFJetsTauOverlapRemoval);
