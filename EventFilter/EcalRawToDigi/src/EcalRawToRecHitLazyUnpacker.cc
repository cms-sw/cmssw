#include "EventFilter/EcalRawToDigi/interface/EcalRawToRecHitLazyUnpacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalRawToRecHitLazyUnpacker::EcalRawToRecHitLazyUnpacker(const EcalRegionCabling & cable,
							 const EcalUnpackerWorker & worker,
							 const FEDRawDataCollection& fedcollection):
  raw_(&fedcollection), cabling_(&cable), worker_(&worker)
{
  LogDebug("EcalRawToRecHit|LazyUnpacker")<<"lazy unpacker created with a max of: "
					  <<FEDNumbering::MAXECALFEDID-FEDNumbering::MINECALFEDID+1
					  <<" regions";
}

EcalRawToRecHitLazyUnpacker::~EcalRawToRecHitLazyUnpacker(){
  //clear the cache to avoid memory leak
}
void EcalRawToRecHitLazyUnpacker::fill(const uint32_t & i, record_type & rec){
  LogDebug("EcalRawToRecHit|LazyUnpacker")<<"filling for index: "<<i;

  std::map<uint32_t, std::auto_ptr<EcalRecHitCollection> > ::iterator f= cachedRecHits.find(i);
  if (f==cachedRecHits.end()){
    LogDebug("EcalRawToRecHit|LazyUnpacker")<<"needs to be unpacked.";
    //need to unpack

    LogDebug("EcalRawToRecHit|LazyUnpacker")<<"calling the worker to work on that index: "<<i;
    std::auto_ptr< EcalRecHitCollection > rechits = worker_->work(i, *raw_);

    LogDebug("EcalRawToRecHit|LazyUnpacker")<<"inserting: "<<rechits->size() <<" rechit(s) in the record.";
    rec.insert(rec.end(), rechits->begin(), rechits->end());
  }
}

