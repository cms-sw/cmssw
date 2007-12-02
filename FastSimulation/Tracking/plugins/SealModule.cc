#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(TrajectorySeedProducer);
