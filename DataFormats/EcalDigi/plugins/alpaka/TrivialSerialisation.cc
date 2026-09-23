#include "DataFormats/EcalDigi/interface/EcalDigiHostCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiPhase2HostCollection.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiDeviceCollection.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiPhase2DeviceCollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(EcalDigiHostCollection, EcalDigiDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(EcalDigiPhase2HostCollection, EcalDigiPhase2DeviceCollection);
