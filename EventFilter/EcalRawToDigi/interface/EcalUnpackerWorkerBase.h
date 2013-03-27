#ifndef EcalUnpackerWorkerBase_H
#define EcalUnpackerWorkerBase_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerRecord.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/EcalRawToDigi/interface/MyWatcher.h"

class  EcalUnpackerWorkerBase {
 public:

  EcalUnpackerWorkerBase(){}
  
  virtual  ~EcalUnpackerWorkerBase(){}
  
  // the method that does it all
  virtual std::auto_ptr<EcalRecHitCollection> work(const uint32_t & i, const FEDRawDataCollection&) const { return std::auto_ptr<EcalRecHitCollection>(0);}
  
  // method to set things up once per event
  virtual void update(const edm::Event & e) const {};
  virtual void setEvent(edm::Event const& e) const {};

  virtual void write(edm::Event &e) const {};

  virtual void setHandles(const EcalUnpackerWorkerRecord & iRecord) {};
  virtual void set(const edm::EventSetup & es) const {};

  virtual unsigned int maxElementIndex() const {return 0;};
};

#endif
