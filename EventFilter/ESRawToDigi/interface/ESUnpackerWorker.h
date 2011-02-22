#ifndef ESUnpackerWorker_H
#define ESUnpackerWorker_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerRecord.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/EcalRawToDigi/interface/MyWatcher.h"
#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerBase.h"

#include "EventFilter/ESRawToDigi/interface/ESUnpacker.h"
#include "RecoLocalCalo/EcalRecProducers/plugins/ESRecHitWorker.h"

//forward declaration. just to be friend
class EcalRawToRecHitByproductProducer;

class ESUnpackerWorker :public EcalUnpackerWorkerBase { 
 public:

  ESUnpackerWorker(const edm::ParameterSet & conf);
  
  ~ESUnpackerWorker();
  
  // the method that does it all
  std::auto_ptr<EcalRecHitCollection> work(const uint32_t & i, const FEDRawDataCollection&) const;
  
  // method to set things up once per event
  void update(const edm::Event & e) const;
  
  void write(edm::Event &e) const;

  void setHandles(const EcalUnpackerWorkerRecord & iRecord);
  void set(const edm::EventSetup & es) const;

  unsigned int maxElementIndex() const { return EcalRegionCabling::maxESElementIndex();}
  
 private:
  
  ESUnpacker* ESUnpacker_;
  ESRecHitWorkerBaseClass * RHWorker_;

};


#endif
