#include "DataFormats/HcalRecHit/interface/HcalRecHitHostCollection.h"
#include "DataFormats/HcalRecHit/interface/alpaka/HcalRecHitDeviceCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(hcal::RecHitHostCollection, hcal::RecHitDeviceCollection);
