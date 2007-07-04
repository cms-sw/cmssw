#ifndef _CSCFRONTIERGAINSCONDITIONS_H
#define _CSCFRONTIERGAINSCONDITIONS_H

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
#include "CalibMuon/CSCCalibration/interface/CSCFrontierGainsMap.h"

class CSCFrontierGainsConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFrontierGainsConditions(const edm::ParameterSet&);
      ~CSCFrontierGainsConditions();

      typedef const  CSCGains * ReturnType;

      ReturnType produceGains(const CSCGainsRcd&);

   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
    
    CSCFrontierGainsMap gains;
};

#endif
