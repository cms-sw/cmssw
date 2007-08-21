#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/L3TrackFinder/interface/MuonRoadTrajectoryBuilder.h"
#include "RecoMuon/L3TrackFinder/interface/MuonRoadTrajectoryBuilderESProducer.h"
#include "RecoMuon/L3TrackFinder/interface/MuonCkfTrajectoryBuilder.h"
#include "RecoMuon/L3TrackFinder/interface/MuonCkfTrajectoryBuilderESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MuonRoadTrajectoryBuilderESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MuonCkfTrajectoryBuilderESProducer);
