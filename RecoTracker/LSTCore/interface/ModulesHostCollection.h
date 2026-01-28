#ifndef RecoTracker_LSTCore_interface_ModulesHostCollection_h
#define RecoTracker_LSTCore_interface_ModulesHostCollection_h

#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using ModulesHostCollection = PortableHostCollection<ModulesSoABlocks>;
}  // namespace lst
#endif
