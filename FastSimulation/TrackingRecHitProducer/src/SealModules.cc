#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "FastSimulation/TrackingRecHitProducer/interface/SiTrackerGaussianSmearingRecHitConverter.h"
#include "FastSimulation/TrackingRecHitProducer/interface/SiPixelGaussianSmearingRecHitConverterAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/SiStripGaussianSmearingRecHitConverterAlgorithm.h"

DEFINE_SEAL_MODULE();

using cms::SiTrackerGaussianSmearingRecHitConverter;
DEFINE_ANOTHER_FWK_MODULE(SiTrackerGaussianSmearingRecHitConverter);

