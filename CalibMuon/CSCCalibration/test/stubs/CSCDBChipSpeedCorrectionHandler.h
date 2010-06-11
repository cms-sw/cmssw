#ifndef CSC_DBCHIPSPEEDCORRECTION_SRC_IMPL_H
#define CSC_DBCHIPSPEEDCORRECTION_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBChipSpeedCorrectionRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibMuon/CSCCalibration/interface/CSCChipSpeedCorrectionDBConditions.h"

namespace popcon
{
  class CSCDBChipSpeedCorrectionImpl : public popcon::PopConSourceHandler<CSCDBChipSpeedCorrection>
    {
      
    public:
      void getNewObjects();
      std::string id() const { return m_name;}
      ~CSCDBChipSpeedCorrectionImpl(); 
      CSCDBChipSpeedCorrectionImpl(const edm::ParameterSet& pset);
      
    private:
      std::string m_name;
      bool isForMC;
      std::string dataCorrFileName;
      float dataOffset;
    };
}
#endif
