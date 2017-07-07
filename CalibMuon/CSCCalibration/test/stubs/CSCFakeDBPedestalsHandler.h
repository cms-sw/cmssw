#ifndef CSC_FAKEDBPEDESTALS_SRC_IMPL_H
#define CSC_FAKEDBPEDESTALS_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBPedestals.h"

namespace popcon
{
  class CSCFakeDBPedestalsImpl : public popcon::PopConSourceHandler<CSCDBPedestals>
    {
      
    public:
      void getNewObjects() override;
      std::string id() const override { return m_name;}
      ~CSCFakeDBPedestalsImpl() override; 
      
      CSCFakeDBPedestalsImpl(const edm::ParameterSet& pset);
      
    private:
      std::string m_name;
    };
}
#endif
