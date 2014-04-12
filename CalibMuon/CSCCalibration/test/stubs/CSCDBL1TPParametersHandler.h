#ifndef CSC_L1TPPARAMETERS_SRC_IMPL_H
#define CSC_L1TPPARAMETERS_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBL1TPParametersRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibMuon/CSCCalibration/interface/CSCDBL1TPParametersConditions.h"

namespace popcon
{
  class CSCDBL1TPParametersImpl : public popcon::PopConSourceHandler<CSCDBL1TPParameters>
    {
      
    public:
      void getNewObjects();
      std::string id() const { return m_name;}
      ~CSCDBL1TPParametersImpl(); 
      
      CSCDBL1TPParametersImpl(const edm::ParameterSet& pset);
  
    private:
      std::string m_name;
    };
}
#endif
