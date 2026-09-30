#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::TrackingRecHitHost, reco::TrackingRecHitsSoACollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::TrackingRecHitsMaskingHost, reco::TrackingRecHitsSoACollection);

