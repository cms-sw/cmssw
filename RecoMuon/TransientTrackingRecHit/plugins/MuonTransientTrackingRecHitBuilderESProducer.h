#ifndef RecoMuon_TransientTrackingRecHit_MuonTransientTrackingRecHitBuilderESProducer_H
#define RecoMuon_TransientTrackingRecHit_MuonTransientTrackingRecHitBuilderESProducer_H

/** \class MuonTransientTrackingRecHitBuilderESProducer
 *  ESProducer for the Muon Transient TrackingRecHit Builder. The Builder can be taken from the 
 *  EventSetup, decoupling the code in which it is used w.r.t. the RecoMuon/TransientTrackingRecHit
 *  lib.
 *
 *  $Date: 2007/04/18 15:08:40 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "FWCore/Framework/interface/ESProducer.h"

#include <boost/shared_ptr.hpp>

namespace edm {class ParameterSet;}

class TransientRecHitRecord;

class MuonTransientTrackingRecHitBuilderESProducer: public edm::ESProducer {
public:
  /// Constructor
  MuonTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet&);

  /// Destructor
  virtual ~MuonTransientTrackingRecHitBuilderESProducer();

  // Operations
  boost::shared_ptr<TransientTrackingRecHitBuilder> produce(const TransientRecHitRecord&);

protected:

private:
};
#endif
