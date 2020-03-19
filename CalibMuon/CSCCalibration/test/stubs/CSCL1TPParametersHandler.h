#ifndef CSC_L1TPPARAMETERS_SRC_IMPL_H
#define CSC_L1TPPARAMETERS_SRC_IMPL_H

#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include "CalibMuon/CSCCalibration/interface/CSCL1TPParametersConditions.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCL1TPParametersRcd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace popcon {
  class CSCL1TPParametersImpl : public popcon::PopConSourceHandler<CSCL1TPParameters> {
  public:
    void getNewObjects();
    std::string id() const { return m_name; }
    ~CSCL1TPParametersImpl();

    CSCL1TPParametersImpl(const edm::ParameterSet &pset);

  private:
    std::string m_name;
  };
}  // namespace popcon
#endif
