#ifndef _CSCFAKECROSSTALKCONDITIONS_H
#define _CSCFAKECROSSTALKCONDITIONS_H

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
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeCrosstalkMap.h"

class CSCFakeCrosstalkConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFakeCrosstalkConditions(const edm::ParameterSet&);
      ~CSCFakeCrosstalkConditions();

      typedef const  CSCcrosstalk * ReturnType;

      ReturnType produceCrosstalk(const CSCcrosstalkRcd&);

   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
    
    CSCFakeCrosstalkMap crosstalk;
};

#endif
