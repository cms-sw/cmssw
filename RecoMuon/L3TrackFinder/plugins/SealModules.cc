#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

#include "RecoMuon/L3TrackFinder/interface/MuonCkfTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilderFactory.h"
#include "RecoMuon/L3TrackFinder/interface/HLTMuonL2SelectorForL3IO.h"
#include "RecoMuon/L3TrackFinder/interface/Phase2HLTMuonSelectorForL3.h"

DEFINE_EDM_VALIDATED_PLUGIN(BaseCkfTrajectoryBuilderFactory, MuonCkfTrajectoryBuilder, "MuonCkfTrajectoryBuilder");
DEFINE_FWK_MODULE(HLTMuonL2SelectorForL3IO);
DEFINE_FWK_MODULE(Phase2HLTMuonSelectorForL3);
