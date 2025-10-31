#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "TrivialSerialisation/Common/interface/SerialiserFactory.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/TestStruct.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostObject<portabletest::TestStruct>);

using PortableHostCollectionTestSoALayout = PortableHostCollection<portabletest::TestSoALayout<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollectionTestSoALayout);

using PortableHostMultiCollectionTestSoALayout2 =
    PortableHostMultiCollection<portabletest::TestSoALayout<128, false>, portabletest::TestSoALayout2<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostMultiCollectionTestSoALayout2);

using PortableHostMultiCollectionTestSoALayout3 = PortableHostMultiCollection<portabletest::TestSoALayout<128, false>,
                                                                              portabletest::TestSoALayout2<128, false>,
                                                                              portabletest::TestSoALayout3<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostMultiCollectionTestSoALayout3);
