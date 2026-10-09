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

  using AoSHostCollectionEvolutionZero = PortableHostCollection<AoSEvolutionZero>;
  using AoSHostCollectionEvolutionOne = PortableHostCollection<AoSEvolutionOne>;
  using AoSHostCollectionEvolutionTwo = PortableHostCollection<AoSEvolutionTwo>;
  using AoSHostCollectionEvolutionThree = PortableHostCollection<AoSEvolutionThree>;
  using AoSHostCollectionEvolutionFour = PortableHostCollection<AoSEvolutionFour>;
  using AoSHostCollectionEvolutionFive = PortableHostCollection<AoSEvolutionFive>;
}  // namespace testmodules

#endif  // HeterogeneousCore_TestModules_interface_SchemaEvolutionHostCollection_h
