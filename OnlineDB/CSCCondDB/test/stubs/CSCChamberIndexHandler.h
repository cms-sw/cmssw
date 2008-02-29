#ifndef CSC_CHAMBERINDEX_SRC_IMPL_H
#define CSC_CHAMBERINDEX_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCChamberIndexRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

namespace popcon
{
  class CSCChamberIndexImpl : public popcon::PopConSourceHandler<CSCChamberIndex>
    {
      
    public:
      void getNewObjects();
      std::string id() const { return m_name;}
      ~CSCChamberIndexImpl(); 
      
      CSCChamberIndexImpl(const edm::ParameterSet& pset);
      
    private:
      std::string m_name;
    };
}
#endif
