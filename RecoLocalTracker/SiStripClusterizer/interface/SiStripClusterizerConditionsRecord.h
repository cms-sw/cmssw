#ifndef RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsRecord_h
#define RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsRecord_h

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"

namespace sistrip {
  class SiStripClusterizerConditionsRecord
      : public edm::eventsetup::DependentRecordImplementation<
            SiStripClusterizerConditionsRecord,
            edm::mpl::Vector<SiStripGainRcd, SiStripNoisesRcd, SiStripQualityRcd>> {};
}  // namespace sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsRecord_h