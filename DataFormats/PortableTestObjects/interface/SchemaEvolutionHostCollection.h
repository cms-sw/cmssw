#ifndef DataFormats_PortableTestObjects_interface_SchemaEvolutionHostCollection_h
#define DataFormats_PortableTestObjects_interface_SchemaEvolutionHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/SchemaEvolutionSoA.h"

namespace portabletest {
  using HostCollectionEvolutionZero = PortableHostCollection<SoAEvolutionZero>;
  using HostCollectionEvolutionOne = PortableHostCollection<SoAEvolutionOne>;
  using HostCollectionEvolutionTwo = PortableHostCollection<SoAEvolutionTwo>;
  using HostCollectionEvolutionThree = PortableHostCollection<SoAEvolutionThree>;
  using HostCollectionEvolutionFour = PortableHostCollection<SoAEvolutionFour>;
}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_SchemaEvolutionHostCollection_h
