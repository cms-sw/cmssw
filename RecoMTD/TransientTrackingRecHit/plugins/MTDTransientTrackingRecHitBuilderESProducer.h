#ifndef RecoMTD_TransientTrackingRecHit_MTDTransientTrackingRecHitBuilderESProducer_H
#define RecoMTD_TransientTrackingRecHit_MTDTransientTrackingRecHitBuilderESProducer_H

/** \class MTDTransientTrackingRecHitBuilderESProducer
 *  ESProducer for the MTD Transient TrackingRecHit Builder. The Builder can be taken from the 
 *  EventSetup, decoupling the code in which it is used w.r.t. the RecoMTD/TransientTrackingRecHit
 *  lib.
 *
 *  \author L. Gray - FNAL
 */

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "FWCore/Framework/interface/ESProducer.h"

#include <memory>

namespace edm {class ParameterSet;}

class TransientRecHitRecord;

class MTDTransientTrackingRecHitBuilderESProducer: public edm::ESProducer {
public:
  /// Constructor
  MTDTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet&);

  /// Destructor
  ~MTDTransientTrackingRecHitBuilderESProducer() override;

  // Operations
  std::unique_ptr<TransientTrackingRecHitBuilder> produce(const TransientRecHitRecord&);

protected:

private:
};
#endif
