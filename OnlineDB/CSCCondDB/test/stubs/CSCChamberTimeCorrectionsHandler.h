#ifndef CSC_CHAMBERTIMECORRECTIONS_SRC_IMPL_H
#define CSC_CHAMBERTIMECORRECTIONS_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCChamberTimeCorrectionsRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberTimeCorrectionsValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

namespace popcon
{
  class CSCChamberTimeCorrectionsImpl : public popcon::PopConSourceHandler<CSCChamberTimeCorrections>
    {
      
    public:
      void getNewObjects();
      std::string id() const { return m_name;}
      ~CSCChamberTimeCorrectionsImpl(); 
      CSCChamberTimeCorrectionsImpl(const edm::ParameterSet& pset);
            
    private:
      std::string m_name;
      bool isForMC;
      float ME11offsetMC;
      float ME11offsetData;
      float nonME11offsetMC;
      float nonME11offsetData;
    };
}
#endif
