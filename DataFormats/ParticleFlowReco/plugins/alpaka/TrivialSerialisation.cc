// Include the Eigen core library before including the SoA definitions
#include <Eigen/Core>

#include "DataFormats/ParticleFlowReco/interface/CaloRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::CaloRecHitHostCollection, reco::CaloRecHitDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::PFClusterHostCollection, reco::PFClusterDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::PFRecHitFractionHostCollection, reco::PFRecHitFractionDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::PFRecHitHostCollection, reco::PFRecHitDeviceCollection);
