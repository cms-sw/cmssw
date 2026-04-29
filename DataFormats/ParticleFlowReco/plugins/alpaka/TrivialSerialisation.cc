// Include the Eigen core library before including the SoA definitions
#include <Eigen/Core>

#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(reco::CaloRecHitHostCollection, reco::CaloRecHitDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(reco::PFClusterHostCollection, reco::PFClusterDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(reco::PFRecHitFractionHostCollection,
                                             reco::PFRecHitFractionDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(reco::PFRecHitHostCollection, reco::PFRecHitDeviceCollection);
