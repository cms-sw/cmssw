#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/Portable/interface/PortableHostObjectReadRules.h"
#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestHostObject.h"

#include "DataFormats/PortableTestObjects/interface/ParticleHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/ImageHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/LogitsHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/SimpleNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MultiHeadNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MaskHostCollection.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(portabletest::TestHostCollection);
SET_PORTABLEHOSTMULTICOLLECTION_READ_RULES(portabletest::TestHostMultiCollection2);
SET_PORTABLEHOSTMULTICOLLECTION_READ_RULES(portabletest::TestHostMultiCollection3);
SET_PORTABLEHOSTOBJECT_READ_RULES(portabletest::TestHostObject);

SET_PORTABLEHOSTCOLLECTION_READ_RULES(portabletest::ParticleHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(portabletest::SimpleNetHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(portabletest::MultiHeadNetHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(portabletest::ImageHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(portabletest::LogitsHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(portabletest::MaskHostCollection);
