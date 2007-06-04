#ifndef _CSCFAKENOISEMATRIXCONDITIONS_H
#define _CSCFAKENOISEMATRIXCONDITIONS_H

#include <memory>
#include "FWCore/Framework/interface/SourceFactory.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeNoiseMatrixMap.h"

class CSCFakeNoiseMatrixConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFakeNoiseMatrixConditions(const edm::ParameterSet&);
      ~CSCFakeNoiseMatrixConditions();

      typedef const  CSCNoiseMatrix * ReturnType;

      ReturnType produceNoiseMatrix(const CSCNoiseMatrixRcd&);

   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
    
    CSCFakeNoiseMatrixMap matrix;
};

#endif
