#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(portabletest::TestDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(portabletest::TestDeviceCollection2);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(portabletest::TestDeviceCollection3);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(portabletest::TestDeviceObject);
