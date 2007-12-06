#include "EventFilter/EcalRawToDigi/interface/EcalRawToRecHitLazyUnpacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalRawToRecHitLazyUnpacker::EcalRawToRecHitLazyUnpacker(const EcalRegionCabling & cable,
							 const EcalUnpackerWorker & worker,
							 const FEDRawDataCollection& fedcollection):
  Base(EcalRegionCabling::maxElementIndex()), raw_(&fedcollection), cabling_(&cable), worker_(&worker)
{
  LogDebug("EcalRawToRecHit|LazyUnpacker")<<"lazy unpacker created with a max of: "
					  <<FEDNumbering::getEcalFEDIds().second-FEDNumbering::getEcalFEDIds().first+1
					  <<" regions";
}

EcalRawToRecHitLazyUnpacker::~EcalRawToRecHitLazyUnpacker(){
  //clear the cache to avoid memory leak
}
void EcalRawToRecHitLazyUnpacker::fill( uint32_t & i){
  LogDebug("EcalRawToRecHit|LazyUnpacker")<<"filling for index: "<<i;

  std::map<uint32_t, std::auto_ptr<EcalRecHitCollection> > ::iterator f= cachedRecHits.find(i);
  if (f==cachedRecHits.end()){
    LogDebug("EcalRawToRecHit|LazyUnpacker")<<"needs to be unpacked.";
    //need to unpack

    LogDebug("EcalRawToRecHit|LazyUnpacker")<<"calling the worker to work on that index: "<<i;
    std::auto_ptr< EcalRecHitCollection > rechits = worker_->work(i, *raw_);

    LogDebug("EcalRawToRecHit|LazyUnpacker")<<"inserting: "<<rechits->size() <<" rechit(s) in the record.";
    Base::record_type & rec = record();
    rec.insert(rec.end(), rechits->begin(), rechits->end());
  }
}

