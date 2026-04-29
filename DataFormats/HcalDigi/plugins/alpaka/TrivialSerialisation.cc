#include "DataFormats/HcalDigi/interface/alpaka/HcalDigiDeviceCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(hcal::Phase0DigiHostCollection, hcal::Phase0DigiDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(hcal::Phase1DigiHostCollection, hcal::Phase1DigiDeviceCollection);
