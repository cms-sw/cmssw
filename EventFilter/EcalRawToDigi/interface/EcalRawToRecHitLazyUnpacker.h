#ifndef EcalRawToRecHitLazyUnpacker_H
#define EcalRawToRecHitLazyUnpacker_H

#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerBase.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"
 
#include "DataFormats/EcalRecHit/interface/EcalRecHitComparison.h"

class EcalRawToRecHitLazyUnpacker : public edm::LazyUnpacker<EcalRecHit> {
 public:

  typedef edm::DetSet<EcalRecHit> DetSet;
  typedef edm::LazyUnpacker<EcalRecHit> Base;

  EcalRawToRecHitLazyUnpacker(const EcalRegionCabling & cable,
			      const EcalUnpackerWorkerBase & worker,
			      const FEDRawDataCollection& fedcollection);
  
  virtual ~EcalRawToRecHitLazyUnpacker();

  // mandatory for actual unpacking stuff
  virtual void fill(const uint32_t&, record_type &);

 private:

  // This a temporary fix for the bug caused by the
  // bad design in EcalUnpackerWorker. It should be
  // removed as soon as no longer necessary.
  virtual void setEvent(edm::Event const& e) const { worker_->setEvent(e); }

  const FEDRawDataCollection* raw_;

  const EcalRegionCabling* cabling_;

  const EcalUnpackerWorkerBase* worker_;

};


#endif
