// 
/**\class L1GTDigiToRaw

 Description: generate raw data from digis - for testing pouposes

*/
//
//         Author:  Ivan Mikulec
//         Created:  Fri Sep 29 17:10:49 CEST 2006
//
#ifndef EventFilter_L1GlobalTriggerRawToDigi_L1GTDigiToRaw_h
#define EventFilter_L1GlobalTriggerRawToDigi_L1GTDigiToRaw_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//
class FEDRawDataCollection;
class L1MuGMTReadoutRecord;
class L1MuGMTReadoutCollection;

using namespace std;

class L1GTDigiToRaw : public edm::EDProducer {
   public:
      explicit L1GTDigiToRaw(const edm::ParameterSet&);
      ~L1GTDigiToRaw();

   private:
      virtual void beginJob(const edm::EventSetup&) {}
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() {}

      void pack(L1MuGMTReadoutCollection const*, auto_ptr<FEDRawDataCollection>&);
      unsigned packGTFE(unsigned char*);
      unsigned packGMT(L1MuGMTReadoutRecord const&, unsigned char*);

      // ----------member data ---------------------------

};

#endif // EventFilter_L1GlobalTriggerRawToDigi_L1GTDigiToRaw_h
