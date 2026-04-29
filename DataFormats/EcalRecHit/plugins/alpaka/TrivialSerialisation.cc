#include "DataFormats/EcalRecHit/interface/alpaka/EcalRecHitDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(EcalRecHitHostCollection, EcalRecHitDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(EcalUncalibratedRecHitHostCollection,
                                             EcalUncalibratedRecHitDeviceCollection);
