#ifndef RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsHost_h
#define RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsSoA.h"

namespace sistrip {
  using SiStripClusterizerConditionsHost = PortableHostMultiCollection<SiStripClusterizerConditionsDetToFedsSoA,
                                                                       SiStripClusterizerConditionsData_fedchSoA,
                                                                       SiStripClusterizerConditionsData_stripSoA,
                                                                       SiStripClusterizerConditionsData_apvSoA>;
  using SiStripClusterizerConditionsDetToFedsHost = PortableHostCollection<SiStripClusterizerConditionsDetToFedsSoA>;
}  // namespace sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsHost_h
