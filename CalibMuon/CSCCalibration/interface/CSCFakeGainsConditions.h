#ifndef _CSCFAKEGAINSCONDITIONS_H
#define _CSCFAKEGAINSCONDITIONS_H

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

#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCFakeGainsConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFakeGainsConditions(const edm::ParameterSet&);
      ~CSCFakeGainsConditions();

      float mean,min,minchi;
      int seed;long int M;

      void prefillGains(); 

      typedef const  CSCGains * ReturnType;

      ReturnType produceGains(const CSCGainsRcd&);

   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
    CSCGains *cngains ;

};

#endif
