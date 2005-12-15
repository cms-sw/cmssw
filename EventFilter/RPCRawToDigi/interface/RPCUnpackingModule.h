#ifndef EVENTFILTER_RPCUnpackingModule_H
#define EVENTFILTER_RPCUnpackingModule_H


/** \class RPCUnpackingModule
 *  Driver class for unpacking RPC raw data (DCC format)
 *
 *  $Date: 2005/12/14 13:36:29 $
 *  $Revision: 1.6 $
 *  \author Ilaria Segoni - CERN
 */


#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>
#include <EventFilter/RPCRawToDigi/interface/RPCEventData.h>

#include <iostream>

class RPCMonitorInterface;

class RPCUnpackingModule: public edm::EDProducer {
  public:
    
    ///Constructor
    RPCUnpackingModule(const edm::ParameterSet& pset);
    
    ///Destructor
    virtual ~RPCUnpackingModule();
 
   /** Retrieves a RPCDigiCollection from the Event, creates a
      FEDRawDataCollection (EDProduct) using the DigiToRaw converter,
      and attaches it to the Event. */
    void produce(edm::Event & e, const edm::EventSetup& c); 
  
    /// Unpacks FED Header(s), returns number of Headers 
    int HeaderUnpacker(const unsigned char* headerIndex);

    /// Unpacks FED Trailer(s), returns number of Trailers 
    int TrailerUnpacker(const unsigned char* trailererIndex);
    
    /// Record Unpacker 
    void recordUnpack(RPCRecord::recordTypes  type , const unsigned char* recordIndex); 
    
    ///Fills Container for DQM imformation
    RPCEventData eventData(){return rpcData;}


  private:
  
    bool printout;
    bool hexprintout;  
    RPCEventData rpcData; 
    int nEvents;
    int currentBX;
    int currentChn;
    
    bool instatiateDQM;   
    RPCMonitorInterface * monitor;

  };


#endif
