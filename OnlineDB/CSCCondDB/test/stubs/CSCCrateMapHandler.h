#ifndef CSC_CRATE_MAP_SRC_IMPL_H
#define CSC_CRATE_MAP_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrateMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

namespace popcon {
  class CSCCrateMapImpl : public popcon::PopConSourceHandler<CSCCrateMap> {
  public:
    void getNewObjects();
    std::string id() const { return m_name; }
    ~CSCCrateMapImpl();
    CSCCrateMapImpl(const edm::ParameterSet& pset);

  private:
    std::string m_name;
  };
}  // namespace popcon
#endif
