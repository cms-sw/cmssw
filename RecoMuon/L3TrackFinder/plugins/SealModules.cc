#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/L3TrackFinder/interface/MuonRoadTrajectoryBuilder.h"
#include "RecoMuon/L3TrackFinder/interface/MuonRoadTrajectoryBuilderESProducer.h"
#include "RecoMuon/L3TrackFinder/interface/MuonCkfTrajectoryBuilder.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"


DEFINE_FWK_EVENTSETUP_MODULE(MuonRoadTrajectoryBuilderESProducer);

#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilderFactory.h"

DEFINE_EDM_PLUGIN(BaseCkfTrajectoryBuilderFactory, MuonCkfTrajectoryBuilder, "MuonCkfTrajectoryBuilder");
