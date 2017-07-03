#ifndef CSC_DBNOISEMATRIX_SRC_IMPL_H
#define CSC_DBNOISEMATRIX_SRC_IMPL_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixDBConditions.h"

namespace popcon
{
  class CSCDBNoiseMatrixImpl : public popcon::PopConSourceHandler<CSCDBNoiseMatrix>
    {
      
    public:
      void getNewObjects() override;
      std::string id() const override { return m_name;}
      ~CSCDBNoiseMatrixImpl() override; 
      CSCDBNoiseMatrixImpl(const edm::ParameterSet& pset);
      
    private:
      std::string m_name;
    };
}
#endif
