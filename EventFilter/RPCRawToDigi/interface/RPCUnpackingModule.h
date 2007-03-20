#ifndef RPCUnpackingModule_H
#define RPCUnpackingModule_H


/** \class RPCUnpackingModule
 *  Driver class for unpacking RPC raw data (DCC format)
 *
 *  $Date: 2006/03/30 15:15:42 $
 *  $Revision: 1.11 $
 *  \author Ilaria Segoni - CERN
 */

#include "FWCore/Framework/interface/EDProducer.h"
namespace edm { class Event; class EventSetup; }


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
  
private:
  unsigned long eventCounter_;
};


#endif
