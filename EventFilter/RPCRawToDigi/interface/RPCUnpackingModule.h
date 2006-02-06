#ifndef EVENTFILTER_RPCUnpackingModule_H
#define EVENTFILTER_RPCUnpackingModule_H


/** \class RPCUnpackingModule
 *  Driver class for unpacking RPC raw data (DCC format)
 *
 *  $Date: 2005/12/15 17:47:59 $
 *  $Revision: 1.7 $
 *  \author Ilaria Segoni - CERN
 */


#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>
#include <EventFilter/RPCRawToDigi/interface/RPCEventData.h>

#include <iostream>
#include <vector>

class RPCDetId;

using namespace std;


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
    int unpackHeader(const unsigned char* headerIndex) const;

    /// Unpacks FED Trailer(s), returns number of Trailers 
    int unpackTrailer(const unsigned char* trailererIndex) const;
    
    ///Unpack record type Start of BX Data and return BXN
    int unpackBXRecord(const unsigned char* recordIndex); 
      
    ///Unpack record type Channel Data and return DetId
    RPCDetId unpackChannelRecord(const unsigned char* recordIndex); 
      
    ///Unpack record type Chamber Data and return vector of Strip ID
    vector<int> unpackChamberRecord(const unsigned char* recordIndex); 
    
    ///Unpack RMB corrupted/discarded data
    void unpackRMBCorruptedRecord(const unsigned char* recordIndex);
    
    
          
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
