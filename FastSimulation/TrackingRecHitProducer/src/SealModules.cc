#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

//--- Our components which we want Framework to know about:                                                                    
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "FastSimulation/TrackingRecHitProducer/interface/FastStripCPEESProducer.h"
#include "FastSimulation/TrackingRecHitProducer/interface/FastPixelCPEESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FastSimulation/TrackingRecHitProducer/interface/SiClusterTranslator.h"
#include "FastSimulation/TrackingRecHitProducer/interface/SiTrackerGaussianSmearingRecHitConverter.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitTranslator.h"

#include "FastSimulation/TrackingRecHitProducer/interface/FastPixelCPE.h"
#include "FastSimulation/TrackingRecHitProducer/interface/FastStripCPE.h"

EVENTSETUP_DATA_REG(FastPixelCPE);
EVENTSETUP_DATA_REG(FastStripCPE);

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(SiTrackerGaussianSmearingRecHitConverter);
DEFINE_ANOTHER_FWK_MODULE(TrackingRecHitTranslator);
DEFINE_ANOTHER_FWK_MODULE(SiClusterTranslator);
