#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FastSimulation/TrackingRecHitProducer/interface/SiTrackerGaussianSmearingRecHitConverter.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitTranslator.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(SiTrackerGaussianSmearingRecHitConverter);
DEFINE_ANOTHER_FWK_MODULE(TrackingRecHitTranslator);

