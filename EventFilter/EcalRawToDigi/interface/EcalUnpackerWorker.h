#ifndef EcalUnpackerWorker_H
#define EcalUnpackerWorker_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerRecord.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerBaseClass.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "EventFilter/EcalRawToDigi/interface/MyWatcher.h"

#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerBase.h"

//forward declaration. just to be friend
class EcalRawToRecHitByproductProducer;

class EcalUnpackerWorker : public EcalUnpackerWorkerBase {
 public:

  EcalUnpackerWorker(const edm::ParameterSet & conf);
  
  ~EcalUnpackerWorker();
  
  // the method that does it all
  std::auto_ptr<EcalRecHitCollection> work(const uint32_t & i, const FEDRawDataCollection&) const;
  
  // method to set things up once per event
  void update(const edm::Event & e) const;
  void setEvent(edm::Event const& e) const {evt = &e;}

  void write(edm::Event &e) const;

  void setHandles(const EcalUnpackerWorkerRecord & iRecord);
  void set(const edm::EventSetup & es) const;

  unsigned int maxElementIndex() const { return EcalRegionCabling::maxElementIndex();}
  
 private:

  // This is bad design. EventSetup data types and Event product types
  // should not contain a pointer to the Event. The Event object has a
  // lifetime of one module and a pointer to it should not be saved in
  // these types. This is very fragile. This code needs to be redesigned
  // to remove this pointer entirely.
  mutable const edm::Event * evt;

  DCCDataUnpacker * unpacker_;

  EcalElectronicsMapper * myMap_;
  mutable std::auto_ptr<EBDigiCollection> productDigisEB;
  mutable std::auto_ptr<EEDigiCollection> productDigisEE;

  friend class EcalRawToRecHitByproductProducer;
  mutable std::auto_ptr<EcalRawDataCollection> productDccHeaders;
  mutable std::auto_ptr< EBDetIdCollection> productInvalidGains;
  mutable std::auto_ptr< EBDetIdCollection> productInvalidGainsSwitch;
  mutable std::auto_ptr< EBDetIdCollection> productInvalidChIds;
  mutable std::auto_ptr< EEDetIdCollection> productInvalidEEGains;
  mutable std::auto_ptr<EEDetIdCollection> productInvalidEEGainsSwitch;
  mutable std::auto_ptr<EEDetIdCollection> productInvalidEEChIds;
  mutable std::auto_ptr<EBSrFlagCollection> productEBSrFlags;
  mutable std::auto_ptr<EESrFlagCollection> productEESrFlags;
  mutable std::auto_ptr<EcalTrigPrimDigiCollection> productTps;
  mutable std::auto_ptr<EcalPSInputDigiCollection> productPSs;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidTTIds;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidZSXtalIds;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidBlockLengths;
  mutable std::auto_ptr<EcalPnDiodeDigiCollection> productPnDiodeDigis;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemTtIds;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemBlockSizes;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemChIds;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemGains;

  mutable edm::ESHandle<EcalRegionCabling> cabling;

  EcalUncalibRecHitWorkerBaseClass * UncalibWorker_;
  EcalRecHitWorkerBaseClass * CalibWorker_;

 public:

  template <class DID> void work(EcalDigiCollection::const_iterator & beginDigi,
				 EcalDigiCollection::const_iterator & endDigi,
				 std::auto_ptr<EcalUncalibratedRecHitCollection> & uncalibRecHits,
				 std::auto_ptr< EcalRecHitCollection > & calibRechits)const{
//    MyWatcher watcher("<Worker>");
    LogDebug("EcalRawToRecHit|Worker")<<"ready to work on digis.";
//<<watcher.lap();

    EcalDigiCollection::const_iterator itdg = beginDigi;
    /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"iterator check." ;

    for(; itdg != endDigi; ++itdg) 
      {

	//get the uncalibrated rechit
	/*R*/ LogDebug("EcalRawToRecHit|Worker")<<"ready to make Uncalib rechit." ;
//<<watcher.lap();
	if (!UncalibWorker_->run(*evt, itdg, *uncalibRecHits)) continue;
	EcalUncalibratedRecHit & EURH=uncalibRecHits->back();

	/*R*/ LogDebug("EcalRawToRecHit|Worker")<<"creating a rechit." ;
//<<watcher.lap();
	if (!CalibWorker_->run(*evt, EURH, *calibRechits)) continue;
	/*R*/ LogDebug("EcalRawToRecHit|Worker")<<"created." ;
//<<watcher.lap();

      }//loop over digis
  }

};


#endif
