#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"

#ifndef LST_STANDALONE
SET_PORTABLEHOSTMULTICOLLECTION_READ_RULES(lst::LSTInputHostCollection);
#endif
