#include "DataFormats/PortableTestObjects/interface/ImageHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/LogitsHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MaskHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MultiHeadNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/ParticleHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/SimpleNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestHostObject.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::ImageHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::LogitsHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::MaskHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::MultiHeadNetHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::ParticleHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::SimpleNetHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::TestHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::TestHostCollection2);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::TestHostCollection3);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::TestHostObject);
