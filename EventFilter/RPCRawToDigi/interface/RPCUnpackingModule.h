#ifndef RPCUnpackingModule_H
#define RPCUnpackingModule_H


/** \class RPCUnpackingModule
 *  Driver class for unpacking RPC raw data (DCC format)
 *
 *  $Date: 2006/03/30 14:37:00 $
 *  $Revision: 1.10 $
 *  \author Ilaria Segoni - CERN
 */

#include <EventFilter/RPCRawToDigi/interface/RPCFEDData.h>

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>


class RPCDetId;


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
    int unpackHeader(const unsigned char* headerIndex, RPCFEDData & rawData);

    /// Unpacks FED Trailer(s), returns number of Trailers 
    int unpackTrailer(const unsigned char* trailererIndex, RPCFEDData & rawData);
    
          
  private:
  
    bool printout;
    int nEvents;
    
    bool instatiateDQM;   
    RPCMonitorInterface * monitor;

  };


#endif
