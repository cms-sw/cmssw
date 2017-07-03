#ifndef CSC_L1TPPARAMETERS_SRC_IMPL_H
#define CSC_L1TPPARAMETERS_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCL1TPParametersRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibMuon/CSCCalibration/interface/CSCL1TPParametersConditions.h"

namespace popcon
{
  class CSCL1TPParametersImpl : public popcon::PopConSourceHandler<CSCL1TPParameters>
    {
      
    public:
      void getNewObjects() override;
      std::string id() const override { return m_name;}
      ~CSCL1TPParametersImpl() override; 
      
      CSCL1TPParametersImpl(const edm::ParameterSet& pset);
  
    private:
      std::string m_name;
    };
}
#endif
