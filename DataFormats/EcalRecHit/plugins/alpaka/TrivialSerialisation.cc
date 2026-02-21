#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactory.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(EcalUncalibratedRecHitDeviceCollection,
                                          "EcalUncalibratedRecHitDeviceCollection");
