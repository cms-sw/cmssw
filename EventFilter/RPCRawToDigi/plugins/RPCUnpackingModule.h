#ifndef RPCUnpackingModule_H
#define RPCUnpackingModule_H


/** \class RPCUnpackingModule
 *  Driver class for unpacking RPC raw data (DCC format)
 *
 *  $Date: 2008/03/11 16:23:47 $
 *  $Revision: 1.3 $
 *  \author Ilaria Segoni - CERN
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

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
    void produce(edm::Event & ev, const edm::EventSetup& es); 
  
private:
  edm::InputTag dataLabel_;
  unsigned long eventCounter_;
  const RPCReadOutMapping* theCabling;
};


#endif
