#ifndef EVENTFILTER_RPCUnpackingModule_H
#define EVENTFILTER_RPCUnpackingModule_H


#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <EventFilter/RPCRawToDigi/interface/RPCEventData.h>

#include <iostream>


class RPCUnpackingModule: public edm::EDProducer {
  public:
    
    RPCUnpackingModule(const edm::ParameterSet& pset);
    virtual ~RPCUnpackingModule();

  /** Retrieves a RPCDigiCollection from the Event, creates a
      FEDRawDataCollection (EDProduct) using the DigiToRaw converter,
      and attaches it to the Event. */
   void produce(edm::Event & e, const edm::EventSetup& c); 


   
   RPCEventData eventData(){return rpcData;}

  private:
  
  RPCEventData rpcData; 
   

  };

DEFINE_FWK_MODULE(RPCUnpackingModule)

#endif
