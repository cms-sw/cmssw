#ifndef Calibration_EcalCalibAlgos_classes_h
#define Calibration_EcalCalibAlgos_classes_h

#include "DataFormats/Common/interface/Wrapper.h"

#include "Calibration/EcalCalibAlgos/interface/EcalPhiSymRecHit.h"
#include "Calibration/EcalCalibAlgos/interface/EcalPhiSymInfo.h"

namespace {
  struct dictionary {
    EcalPhiSymRecHit dummy11;
    std::vector<EcalPhiSymRecHit> dummy12;
    edm::Wrapper<EcalPhiSymRecHitCollection> dummy13;

    EcalPhiSymInfo dummy21;
    std::vector<EcalPhiSymInfo> dummy22;
    edm::Wrapper<EcalPhiSymInfo> dummy23;
  };
}  // namespace

#endif
