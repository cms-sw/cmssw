#ifndef CSC_FAKEDBGAINS_SRC_IMPL_H
#define CSC_FAKEDBGAINS_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBCrosstalk.h"

namespace popcon
{
  class CSCFakeDBCrosstalkImpl : public popcon::PopConSourceHandler<CSCDBCrosstalk>
    {
      
    public:
      void getNewObjects();
      std::string id() const { return m_name;}
      ~CSCFakeDBCrosstalkImpl(); 
      
      CSCFakeDBCrosstalkImpl(const edm::ParameterSet& pset);
      
    private:
      std::string m_name;
    };
}
#endif
