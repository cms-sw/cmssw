#ifndef EcalRawToRecHitLazyUnpacker_H
#define EcalRawToRecHitLazyUnpacker_H

#include "DataFormats/EcalRecHit/interface/LazyGetter.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorker.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"
 
#include "DataFormats/EcalRecHit/interface/EcalRecHitComparison.h"

class EcalRawToRecHitLazyUnpacker : public edm::LazyUnpacker<EcalRecHit> {
 public:

  typedef edm::DetSet<EcalRecHit> DetSet;
  typedef edm::LazyUnpacker<EcalRecHit> Base;

  EcalRawToRecHitLazyUnpacker(const EcalRegionCabling & cable,
			      const EcalUnpackerWorker & worker,
			      const FEDRawDataCollection& fedcollection);
  
  virtual ~EcalRawToRecHitLazyUnpacker();

  // mandatory for actual unpacking stuff
  virtual void fill(const uint32_t&, record_type &);

 private:
  
  const FEDRawDataCollection* raw_;

  const EcalRegionCabling* cabling_;

  const EcalUnpackerWorker* worker_;

  //cache
  std::map<uint32_t, std::auto_ptr<EcalRecHitCollection> > cachedRecHits;
};


#endif
