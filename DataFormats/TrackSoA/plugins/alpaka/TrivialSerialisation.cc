#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactory.h"

DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(ALPAKA_ACCELERATOR_NAMESPACE::reco::TracksSoACollection,
                                          "reco::TracksDeviceCollection");
