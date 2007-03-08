#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FastSimulation/Tracking/interface/GSTrackCandidateMaker.h"

//#include "FWCore/Framework/interface/EventSetup.h"
//#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

using cms::GSTrackCandidateMaker;

EVENTSETUP_DATA_REG(TrackerTrajectoryBuilder);
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(GSTrackCandidateMaker);
