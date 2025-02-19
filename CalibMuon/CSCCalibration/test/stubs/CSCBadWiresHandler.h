#ifndef CSC_BADWIRES_SRC_IMPL_H
#define CSC_BADWIRES_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCBadWiresRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibMuon/CSCCalibration/interface/CSCBadWiresConditions.h"

namespace popcon
{
  class CSCBadWiresImpl : public popcon::PopConSourceHandler<CSCBadWires>
    {
      
    public:
      void getNewObjects();
      std::string id() const { return m_name;}
      ~CSCBadWiresImpl(); 
      
      CSCBadWiresImpl(const edm::ParameterSet& pset);
      
    private:
      std::string m_name;
    };
}
#endif
