#ifndef DataFormats_PortableTestObjects_interface_SchemaEvolutionHostCollection_h
#define DataFormats_PortableTestObjects_interface_SchemaEvolutionHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/SchemaEvolutionSoA.h"

namespace portabletest {
    using SchemaEvolutionHostCollection = PortableHostCollection<SchemaEvolutionSoA>;
}

#endif  // DataFormats_PortableTestObjects_interface_SchemaEvolutionHostCollection_h
