#ifndef _CSCFAKEDBPEDESTALS_H
#define _CSCFAKEDBPEDESTALS_H

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

#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCFakeDBPedestals: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFakeDBPedestals(const edm::ParameterSet&);
      ~CSCFakeDBPedestals();

      float meanped,meanrms;
      int seed;long int M;
      
      void prefillDBPedestals();

      typedef const  CSCDBPedestals * ReturnType;

      ReturnType produceDBPedestals(const CSCDBPedestalsRcd&);

   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
      CSCDBPedestals *cndbpedestals ;   
};

#endif
