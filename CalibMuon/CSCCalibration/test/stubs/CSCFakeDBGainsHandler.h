#ifndef CSC_FAKEDBGAINS_SRC_IMPL_H
#define CSC_FAKEDBGAINS_SRC_IMPL_H

#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include "CalibMuon/CSCCalibration/interface/CSCFakeDBGains.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace popcon {
  class CSCFakeDBGainsImpl : public popcon::PopConSourceHandler<CSCDBGains> {
  public:
    void getNewObjects();
    std::string id() const { return m_name; }
    ~CSCFakeDBGainsImpl();

    CSCFakeDBGainsImpl(const edm::ParameterSet &pset);

  private:
    std::string m_name;
  };
}  // namespace popcon
#endif
