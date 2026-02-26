#include <Eigen/Core>

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactory.h"

DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(reco::PFRecHitDeviceCollection, "reco::PFRecHitDeviceCollection");

DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(reco::PFClusterDeviceCollection, "reco::PFClusterDeviceCollection");

DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(reco::PFRecHitFractionDeviceCollection,
                                          "reco::PFRecHitFractionDeviceCollection");
