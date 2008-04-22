#ifndef RPCUnpackingModule_H
#define RPCUnpackingModule_H


/** \class RPCUnpackingModule
 *  Driver class for unpacking RPC raw data (DCC format)
 *
 *  $Date: 2007/04/20 15:41:48 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni - CERN
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

namespace edm { class Event; class EventSetup; }
static RPCReadOutMapping* RPCCabling;

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
  edm::InputTag dataLabel_;
  unsigned long eventCounter_;
};


#endif
