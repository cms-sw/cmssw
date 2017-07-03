#ifndef CSC_BADSTRIPS_SRC_IMPL_H
#define CSC_BADSTRIPS_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCBadStripsRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibMuon/CSCCalibration/interface/CSCBadStripsConditions.h"

namespace popcon
{
  class CSCBadStripsImpl : public popcon::PopConSourceHandler<CSCBadStrips>
    {
      
    public:
      void getNewObjects() override;
      std::string id() const override { return m_name;}
      ~CSCBadStripsImpl() override; 
      
      CSCBadStripsImpl(const edm::ParameterSet& pset);
      
    private:
      std::string m_name;
    };
}
#endif
