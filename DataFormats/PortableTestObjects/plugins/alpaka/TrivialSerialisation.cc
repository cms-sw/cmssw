#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactory.h"

DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection,
                                          "portabletest::TestDeviceCollection");
DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection2,
                                          "portabletest::TestDeviceCollection2");
DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection3,
                                          "portabletest::TestDeviceCollection3");
DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceObject,
                                          "portabletest::TestDeviceObject");
