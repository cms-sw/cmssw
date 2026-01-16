// Include the Eigen core library before including the SoA definitions
#include <Eigen/Core>

#include "DataFormats/ParticleFlowReco/interface/CaloRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::CaloRecHitHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::PFClusterHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::PFRecHitFractionHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::PFRecHitHostCollection);
