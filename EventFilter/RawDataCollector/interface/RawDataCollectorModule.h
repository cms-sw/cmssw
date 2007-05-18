#ifndef RawDataCollectorModule_H
#define RawDataCollectorModule_H


/** \class RawDataCollectorModule
 *  Driver class for packing RPC digi data 
 *
 */

#include <EventFilter/RPCRawToDigi/interface/RPCFEDData.h>

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>


class RPCDetId;


class RPCMonitorInterface;

class RawDataCollectorModule: public edm::EDProducer {
  public:
    
    ///Constructor
    RawDataCollectorModule(const edm::ParameterSet& pset);
    
    ///Destructor
    virtual ~RawDataCollectorModule();
 
    void produce(edm::Event & e, const edm::EventSetup& c); 
          
  private:
  
  };


#endif
