#include "DataFormats/PortableTestObjects/interface/alpaka/ImageDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/LogitsDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/MaskDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/MultiHeadNetDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/SimpleNetDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(portabletest::ImageHostCollection, portabletest::ImageDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(portabletest::LogitsHostCollection, portabletest::LogitsDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(portabletest::MaskHostCollection, portabletest::MaskDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(portabletest::MultiHeadNetHostCollection,
                                             portabletest::MultiHeadNetDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(portabletest::ParticleHostCollection,
                                             portabletest::ParticleDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(portabletest::SimpleNetHostCollection,
                                             portabletest::SimpleNetDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(portabletest::TestHostCollection, portabletest::TestDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(portabletest::TestHostCollection2, portabletest::TestDeviceCollection2);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(portabletest::TestHostCollection3, portabletest::TestDeviceCollection3);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(portabletest::TestHostObject, portabletest::TestDeviceObject);
