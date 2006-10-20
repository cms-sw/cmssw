// 
/**\class L1GlobalTriggerRawToDigi

 Description: generate raw data from digis - for testing pouposes

*/
//
//         Author:  Ivan Mikulec
//         Created:  Fri Sep 29 17:10:49 CEST 2006
//
#ifndef EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerRawToDigi_h
#define EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerRawToDigi_h

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
class L1MuGMTReadoutCollection;

using namespace std;

class L1GlobalTriggerRawToDigi : public edm::EDProducer {
   public:
      explicit L1GlobalTriggerRawToDigi(const edm::ParameterSet&);
      ~L1GlobalTriggerRawToDigi();

   private:
      virtual void beginJob(const edm::EventSetup&) {}
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() {}

      void unpackGMT(const unsigned char*, auto_ptr<L1MuGMTReadoutCollection>&);

      // ----------member data ---------------------------

};

#endif // EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerRawToDigi_h
