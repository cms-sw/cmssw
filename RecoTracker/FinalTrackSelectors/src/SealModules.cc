
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/FinalTrackSelectors/interface/SimpleTrackListMerger.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackMultiSelector.h"
#include "RecoTracker/FinalTrackSelectors/interface/AnalyticalTrackSelector.h"
#include "RecoTracker/FinalTrackSelectors/interface/CosmicTrackSelector.h" 


using cms::SimpleTrackListMerger;
using reco::modules::TrackMultiSelector;
using reco::modules::AnalyticalTrackSelector;
using reco::modules::CosmicTrackSelector;


DEFINE_FWK_MODULE(SimpleTrackListMerger);
DEFINE_FWK_MODULE(TrackMultiSelector);
DEFINE_FWK_MODULE(AnalyticalTrackSelector);
DEFINE_FWK_MODULE(CosmicTrackSelector);
