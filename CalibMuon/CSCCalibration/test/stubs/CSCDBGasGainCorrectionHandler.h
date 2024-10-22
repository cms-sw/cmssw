#ifndef CSC_DBCHIPSPEEDCORRECTION_SRC_IMPL_H
#define CSC_DBCHIPSPEEDCORRECTION_SRC_IMPL_H

#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include "CalibMuon/CSCCalibration/interface/CSCGasGainCorrectionDBConditions.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBGasGainCorrectionRcd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace popcon {
  class CSCDBGasGainCorrectionImpl : public popcon::PopConSourceHandler<CSCDBGasGainCorrection> {
  public:
    void getNewObjects();
    std::string id() const { return m_name; }
    ~CSCDBGasGainCorrectionImpl();
    CSCDBGasGainCorrectionImpl(const edm::ParameterSet &pset);

  private:
    std::string m_name;
    bool isForMC;
    std::string dataCorrFileName;
  };
}  // namespace popcon
#endif
