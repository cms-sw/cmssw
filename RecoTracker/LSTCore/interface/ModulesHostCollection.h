#ifndef RecoTracker_LSTCore_interface_ModulesHostCollection_h
#define RecoTracker_LSTCore_interface_ModulesHostCollection_h

#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using ModulesHostCollection = PortableHostMultiCollection<ModulesSoA, ModulesPixelSoA>;
}  // namespace lst
#endif
