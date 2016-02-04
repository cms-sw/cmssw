#ifndef CSC_DBGAINS_SRC_IMPL_H
#define CSC_DBGAINS_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibMuon/CSCCalibration/interface/CSCGainsDBConditions.h"

namespace popcon
{
  class CSCDBGainsImpl : public popcon::PopConSourceHandler<CSCDBGains>
    {
      
    public:
      void getNewObjects();
      std::string id() const { return m_name;}
      ~CSCDBGainsImpl(); 
      
      CSCDBGainsImpl(const edm::ParameterSet& pset);
      
    private:
      std::string m_name;
    };
}
#endif
