#ifndef CSC_DBCHIPSPEEDCORRECTION_SRC_IMPL_H
#define CSC_DBCHIPSPEEDCORRECTION_SRC_IMPL_H

#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include "CalibMuon/CSCCalibration/interface/CSCChipSpeedCorrectionDBConditions.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBChipSpeedCorrectionRcd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace popcon {
  class CSCDBChipSpeedCorrectionImpl : public popcon::PopConSourceHandler<CSCDBChipSpeedCorrection> {
  public:
    void getNewObjects();
    std::string id() const { return m_name; }
    ~CSCDBChipSpeedCorrectionImpl();
    CSCDBChipSpeedCorrectionImpl(const edm::ParameterSet &pset);

  private:
    std::string m_name;
    bool isForMC;
    std::string dataCorrFileName;
    float dataOffset;
  };
}  // namespace popcon
#endif
