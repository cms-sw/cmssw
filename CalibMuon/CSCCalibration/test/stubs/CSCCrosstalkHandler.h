#ifndef CSC_DBCROSSTALK_SRC_IMPL_H
#define CSC_DBCROSSTALK_SRC_IMPL_H

#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkDBConditions.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace popcon {
  class CSCDBCrosstalkImpl : public popcon::PopConSourceHandler<CSCDBCrosstalk> {
  public:
    void getNewObjects();
    std::string id() const { return m_name; }
    ~CSCDBCrosstalkImpl();

    CSCDBCrosstalkImpl(const edm::ParameterSet &pset);

  private:
    std::string m_name;
  };
}  // namespace popcon
#endif
