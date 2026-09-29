#include "DataFormats/PortableTestObjects/interface/ImageHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/LogitsHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MaskHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MultiHeadNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/ParticleHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/SimpleNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestHostObject.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ImageDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/LogitsDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/MaskDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/MultiHeadNetDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/SimpleNetDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(portabletest::ImageHostCollection, portabletest::ImageDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(portabletest::LogitsHostCollection, portabletest::LogitsDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(portabletest::MaskHostCollection, portabletest::MaskDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(portabletest::MultiHeadNetHostCollection,
                                          portabletest::MultiHeadNetDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(portabletest::ParticleHostCollection, portabletest::ParticleDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(portabletest::SimpleNetHostCollection,
                                          portabletest::SimpleNetDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(portabletest::TestHostCollection, portabletest::TestDeviceCollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(portabletest::TestHostCollection2, portabletest::TestDeviceCollection2);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(portabletest::TestHostCollection3, portabletest::TestDeviceCollection3);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(portabletest::TestHostObject, portabletest::TestDeviceObject);
