#ifndef _CSCFAKEDBGAINSPOPCON_H
#define _CSCFAKEDBGAINSPOPCON_H

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

#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCFakeDBGainsPopCon: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFakeDBGainsPopCon(const edm::ParameterSet&);
      ~CSCFakeDBGainsPopCon();

      float mean,min,minchi;
      int seed;long int M;

      void prefillDBGains(); 

      typedef const  CSCDBGains * ReturnType;

      ReturnType produceDBGains(const CSCDBGainsRcd&);

   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
    CSCDBGains *cndbgains ;

};

#endif
