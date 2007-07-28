
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/FinalTrackSelectors/interface/SimpleTrackListMerger.h"

using cms::SimpleTrackListMerger;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SimpleTrackListMerger);
