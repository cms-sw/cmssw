#ifndef _CSCFAKEPEDESTALSCONDITIONS_H
#define _CSCFAKEPEDESTALSCONDITIONS_H

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

#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "CondFormats/DataRecord/interface/CSCPedestalsRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCFakePedestalsConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFakePedestalsConditions(const edm::ParameterSet&);
      ~CSCFakePedestalsConditions();

      float meanped,meanrms;
      int seed;long int M;
      
      void prefillPedestals();

      typedef const  CSCPedestals * ReturnType;

      ReturnType producePedestals(const CSCPedestalsRcd&);

   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
      CSCPedestals *cnpedestals ;   
};

#endif
