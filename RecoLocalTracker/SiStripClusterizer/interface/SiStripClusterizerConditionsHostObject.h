#ifndef RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsHostObject_h
#define RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsHostObject_h

#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/Portable/interface/PortableObject.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsStruct.h"

namespace sistrip {
  using SiStripClusterizerConditionsDetToFedsHostObject = PortableHostObject<DetToFeds>;
  using SiStripClusterizerConditionsGainNoiseCalsHostObject = PortableHostObject<GainNoiseCals>;
}  // namespace sistrip

#endif
