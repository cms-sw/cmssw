#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/VectorHitsSoACollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::TrackingRecHitHost, reco::TrackingRecHitsSoACollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::VectorHitsHost, reco::VectorHitsSoACollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::OTRecHitsHost, reco::OTRecHitsSoACollection);
