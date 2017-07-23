#ifndef GEMUnpackingModule_H
#define GEMUnpackingModule_H


/** \class GEMUnpackingModule
 ** unpacking GEM raw data
 **/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondFormats/DataRecord/interface/GEMEMapRcd.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

class GEMReadOutMapping;
namespace edm { class Event; class EventSetup; class Run; }

class GEMUnpackingModule: public edm::EDProducer {
public:
    
    ///Constructor
    GEMUnpackingModule(const edm::ParameterSet& pset);
    
    ///Destructor
    virtual ~GEMUnpackingModule();
 
   /** Retrieves a GEMDigiCollection from the Event, creates a
      FEDRawDataCollection (EDProduct) using the DigiToRaw converter,
      and attaches it to the Event. */
    void produce(edm::Event & ev, const edm::EventSetup& es) override; 

    void beginRun(const edm::Run &run, const edm::EventSetup& es) override;
  
private:
  edm::EDGetTokenT<FEDRawDataCollection> dataLabel_;
  bool doSynchro_; 
  unsigned long eventCounter_;

  edm::ESWatcher<GEMEMapRcd> gemMapWatcher;

};

#endif
