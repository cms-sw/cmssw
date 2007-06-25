#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/GlobalTrackFinder/interface/MuonRSTrajectoryBuilder.h"
#include "RecoMuon/GlobalTrackFinder/interface/MuonRSTrajectoryBuilderESProducer.h"
#include "RecoMuon/GlobalTrackFinder/interface/MuonCkfTrajectoryBuilder.h"
#include "RecoMuon/GlobalTrackFinder/interface/MuonCkfTrajectoryBuilderESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MuonRSTrajectoryBuilderESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MuonCkfTrajectoryBuilderESProducer);
