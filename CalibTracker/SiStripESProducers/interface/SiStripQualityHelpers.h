#ifndef CALIBTRACKER_SISTRIPESPRODUCERS_INTERFACE_SISTRIPQUALITYHELPERS_H
#define CALIBTRACKER_SISTRIPESPRODUCERS_INTERFACE_SISTRIPQUALITYHELPERS_H

#include <memory>
#include "DQMServices/Core/interface/DQMStore.h"

namespace edm {
  class ParameterSet;
}
class SiStripFedCabling;
class SiStripQuality;
class SiStripQualityRcd;

namespace sistrip {
  /**
   * Create a SiStripQuality record from the list of detected Fed errors
   * in the SiStrip/ReadoutView/FedIdVsApvId DQM histogram
   */
  std::unique_ptr<SiStripQuality> badStripFromFedErr(dqm::harvesting::DQMStore::IGetter& dqmStore,
                                                     const SiStripFedCabling& fedCabling,
                                                     float cutoff);
}  // namespace sistrip

#endif  // CALIBTRACKER_SISTRIPESPRODUCERS_INTERFACE_SISTRIPQUALITYHELPERS_H
