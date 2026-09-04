#ifndef HeterogeneousCore_TestModules_interface_SchemaEvolutionHostCollection_h
#define HeterogeneousCore_TestModules_interface_SchemaEvolutionHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"

#include "HeterogeneousCore/TestModules/interface/SchemaEvolutionSoA.h"

namespace testmodules {
  using HostCollectionEvolutionZero = PortableHostCollection<SoAEvolutionZero>;
  using HostCollectionEvolutionOne = PortableHostCollection<SoAEvolutionOne>;
  using HostCollectionEvolutionTwo = PortableHostCollection<SoAEvolutionTwo>;
  using HostCollectionEvolutionThree = PortableHostCollection<SoAEvolutionThree>;
  using HostCollectionEvolutionFour = PortableHostCollection<SoAEvolutionFour>;
  using HostCollectionEvolutionFive = PortableHostCollection<SoAEvolutionFive>;
}  // namespace testmodules

#endif  // HeterogeneousCore_TestModules_interface_SchemaEvolutionHostCollection_h
