#include "DataFormats/GeometryVector/interface/LocalPoint.h" 
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/TrackingRecHit/interface/RecHit1D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitGlobalState.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

namespace DataFormats_TrackingRecHit {
  struct dictionary {
    AlgebraicSymMatrix as;

    std::vector<TrackingRecHit*> v1;
    TrackingRecHitCollection c1;
    TrackingRecHitRef r1;
    InvalidTrackingRecHit i;
    TrackingRecHitRefProd rp1;
    TrackingRecHitRefVector rv1;
    TrackingRecHitRefVector::const_iterator it1;
    std::pair<edm::OwnVector<TrackingRecHit,
                             edm::ClonePolicy<TrackingRecHit> >::const_iterator,
              edm::OwnVector<TrackingRecHit,
                             edm::ClonePolicy<TrackingRecHit> >::const_iterator> pr1;    
    std::unique_ptr<TrackingRecHitRef> ap1;
    edm::Wrapper<TrackingRecHitCollection> w1;
  };
}
