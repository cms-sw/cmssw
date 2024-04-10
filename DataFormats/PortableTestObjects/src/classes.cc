#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/Portable/interface/PortableHostObjectReadRules.h"
#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestHostObject.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(portabletest::TestHostCollection);
SET_PORTABLEHOSTMULTICOLLECTION_READ_RULES(portabletest::TestHostMultiCollection2);
SET_PORTABLEHOSTMULTICOLLECTION_READ_RULES(portabletest::TestHostMultiCollection3);
SET_PORTABLEHOSTOBJECT_READ_RULES(portabletest::TestHostObject);
