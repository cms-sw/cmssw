//
// plugins/SealModules.cc
// Using new EDM plugin manager (V.Chiochia, April 2007)
//
//--- Our components which we want Framework to know about
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"

//--- The CPE ES Producers
#include "FastSimulation/TrackingRecHitProducer/interface/FastPixelCPEESProducer.h"
#include "FastSimulation/TrackingRecHitProducer/interface/FastStripCPEESProducer.h"

//--- The header files for the Framework infrastructure (macros etc):
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"


DEFINE_FWK_EVENTSETUP_MODULE(FastPixelCPEESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(FastStripCPEESProducer);

