#include <Eigen/Core>

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(reco::PFRecHitDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(reco::PFClusterDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(reco::PFRecHitFractionDeviceCollection);
