#ifndef EVENTFILTER_RPCUnpackingModule_H
#define EVENTFILTER_RPCUnpackingModule_H


/** \class RPCUnpackingModule
 *  Driver class for unpacking RPC raw data (DCC format)
 *
 *  $Date: 2006/02/06 09:24:52 $
 *  $Revision: 1.8 $
 *  \author Ilaria Segoni - CERN
 */


#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>

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
    
          
  private:
  
    bool printout;
    bool hexprintout;  
    int nEvents;
    
    bool instatiateDQM;   
    RPCMonitorInterface * monitor;

  };


#endif
