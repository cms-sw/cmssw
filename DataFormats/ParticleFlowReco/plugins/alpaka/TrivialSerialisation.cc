// Include the Eigen core library before including the SoA definitions
#include <Eigen/Core>

#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(reco::CaloRecHitDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(reco::PFClusterDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(reco::PFRecHitFractionDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(reco::PFRecHitDeviceCollection);
