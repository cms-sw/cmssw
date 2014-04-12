#ifndef RPCUnpackingModule_H
#define RPCUnpackingModule_H


/** \class RPCUnpackingModule
 ** unpacking RPC raw data
 **/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "RPCReadOutMappingWithFastSearch.h"


class RPCReadOutMapping;
namespace edm { class Event; class EventSetup; class Run; }

class RPCUnpackingModule: public edm::EDProducer {
public:
    
    ///Constructor
    RPCUnpackingModule(const edm::ParameterSet& pset);
    
    ///Destructor
    virtual ~RPCUnpackingModule();
 
   /** Retrieves a RPCDigiCollection from the Event, creates a
      FEDRawDataCollection (EDProduct) using the DigiToRaw converter,
      and attaches it to the Event. */
    void produce(edm::Event & ev, const edm::EventSetup& es) override; 

    void beginRun(const edm::Run &run, const edm::EventSetup& es) override;
  
private:
  edm::InputTag dataLabel_;
  bool doSynchro_; 
  unsigned long eventCounter_;

  edm::ESWatcher<RPCEMapRcd> theRecordWatcher;
  const RPCReadOutMapping* theCabling;
  RPCReadOutMappingWithFastSearch theReadoutMappingSearch;
};


#endif
