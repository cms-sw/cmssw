#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

//--- Our components which we want Framework to know about:                                                                    
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "FastStripCPEESProducer.h"
#include "FastPixelCPEESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "SiClusterTranslator.h"
#include "SiTrackerGaussianSmearingRecHitConverter.h"
#include "TrackingRecHitTranslator.h"

#include "FastPixelCPE.h"
#include "FastStripCPE.h"

TYPELOOKUP_DATA_REG(FastPixelCPE);
TYPELOOKUP_DATA_REG(FastStripCPE);



DEFINE_FWK_MODULE(SiTrackerGaussianSmearingRecHitConverter);
DEFINE_FWK_MODULE(TrackingRecHitTranslator);
DEFINE_FWK_MODULE(SiClusterTranslator);
