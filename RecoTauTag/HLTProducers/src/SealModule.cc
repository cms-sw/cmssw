#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTauTag/HLTProducers/interface/PFTauToJetProducer.h"
#include "RecoTauTag/HLTProducers/interface/PFJetToCaloProducer.h"
#include "RecoTauTag/HLTProducers/interface/L1HLTJetsMatching.h"
#include "RecoTauTag/HLTProducers/interface/L1HLTTauMatching.h"
#include "RecoTauTag/HLTProducers/interface/L2TauJetsMerger.h"
#include "RecoTauTag/HLTProducers/interface/CaloTowerCreatorForTauHLT.h"
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

DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, TauRegionalPixelSeedGenerator, "TauRegionalPixelSeedGenerator");      
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, TrackingRegionsFromBeamSpotAndL2Tau, "TrackingRegionsFromBeamSpotAndL2Tau");
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, CandidateSeededTrackingRegionsProducer, "CandidateSeededTrackingRegionsProducer");

DEFINE_FWK_MODULE(L2TauJetsMerger);
DEFINE_FWK_MODULE(L1HLTJetsMatching);
DEFINE_FWK_MODULE(L1HLTTauMatching);
DEFINE_FWK_MODULE(CaloTowerCreatorForTauHLT);
DEFINE_FWK_MODULE(PFTauToJetProducer);
DEFINE_FWK_MODULE(PFJetToCaloProducer);
DEFINE_FWK_MODULE(TauJetSelectorForHLTTrackSeeding);
DEFINE_FWK_MODULE(VertexFromTrackProducer);
//DEFINE_FWK_MODULE(L2TauPixelTrackMatch);
DEFINE_FWK_MODULE(HLTPFTauPairLeadTrackDzMatchFilter);
DEFINE_FWK_MODULE(L2TauPixelIsoTagProducer);

