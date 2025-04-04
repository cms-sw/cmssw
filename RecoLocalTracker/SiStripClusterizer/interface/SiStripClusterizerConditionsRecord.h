#ifndef RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsRecord_h
#define RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsRecord_h

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"

namespace sistrip {
  class SiStripClusterizerConditionsDetToFedsRecord
      : public edm::eventsetup::DependentRecordImplementation<SiStripClusterizerConditionsDetToFedsRecord,
                                                              edm::mpl::Vector<SiStripQualityRcd>> {};

  class SiStripClusterizerConditionsDataRecord
      : public edm::eventsetup::DependentRecordImplementation<
            SiStripClusterizerConditionsDataRecord,
            edm::mpl::Vector<SiStripGainRcd, SiStripNoisesRcd, SiStripQualityRcd>> {};
}  // namespace sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_interface_SiStripClusterizerConditionsRecord_h
