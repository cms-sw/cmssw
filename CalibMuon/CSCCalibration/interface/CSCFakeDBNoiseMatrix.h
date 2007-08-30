#ifndef _CSCFAKEDBNOISEMATRIX_H
#define _CSCFAKEDBNOISEMATRIX_H

#include <memory>
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>


class CSCFakeDBNoiseMatrix: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFakeDBNoiseMatrix(const edm::ParameterSet&);
      ~CSCFakeDBNoiseMatrix();
      
      void prefillDBNoiseMatrix();
      
      typedef const  CSCDBNoiseMatrix * ReturnType;
      ReturnType produceDBNoiseMatrix(const CSCDBNoiseMatrixRcd&);

 private:
      void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
      CSCDBNoiseMatrix *cndbmatrix ;
};

#endif
