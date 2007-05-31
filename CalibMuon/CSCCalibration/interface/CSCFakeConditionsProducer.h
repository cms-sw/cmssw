#ifndef _CSCFAKECONDITIONSPRODUCER_H
#define _CSCFAKECONDITIONSPRODUCER_H

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
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"
//#include "CondFormats/DataRecord/interface/CSCIdentifierRcd.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCPedestalsRcd.h"

class CSCFakeConditionsProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFakeConditionsProducer(const edm::ParameterSet&);
      ~CSCFakeConditionsProducer();

      typedef const  CSCobject * ReturnType;

      ReturnType produce(const CSCNoiseMatrixRcd&);
      ReturnType produce(const CSCcrosstalkRcd&);
   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
    
    CSCFakeMap map_;
    
};

#endif
