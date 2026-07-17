#ifndef DataFormats_TrackerRecHit2D_MTDTrackingRecHit_h
#define DataFormats_TrackerRecHit2D_MTDTrackingRecHit_h

/// A 2D TrackerRecHit with time and time error information

#include <cassert>
#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLMergedClusterCollections.h"
#include "DataFormats/Common/interface/Ref.h"

namespace io_v1 {
  class MTDTrackingRecHit : public TrackerSingleRecHit {
  public:
    MTDTrackingRecHit() : TrackerSingleRecHit() {}

    MTDTrackingRecHit(const LocalPoint& p, const LocalError& e, const GeomDet& idet, const FTLClusterRef& objref)
        : TrackerSingleRecHit(p, e, idet, trackerHitRTTI::mipTiming, objref) {}

    MTDTrackingRecHit* clone() const override { return new MTDTrackingRecHit(*this); }

    // things to specialize from BaseTrackerRecHit
    bool isPhase2() const final { return true; }
    void getKfComponents(KfComponentsHolder& holder) const final;

    int dimension() const final { return 2; }

    // constructor accepting a merged-cluster Ref
    using FTLMergedClusterRef = edm::Ref<FTLMergedClusterCollection, FTLMergedCluster>;
    MTDTrackingRecHit(const LocalPoint& p, const LocalError& e, const GeomDet& idet, const FTLMergedClusterRef& objref)
        : TrackerSingleRecHit(p, e, idet, trackerHitRTTI::mipTiming, objref) {}

    //specific timing stuff
    float energy() const { return omniCluster().mtdMergedCluster().energy(); }
    float time() const { return omniCluster().mtdMergedCluster().time(); }
    float timeError() const { return omniCluster().mtdMergedCluster().timeError(); }
  };
}  // namespace io_v1
using MTDTrackingRecHit = io_v1::MTDTrackingRecHit;
// Instantiations and specializations for FTLRecHitRef and reco::CaloClusterPtr
#include "DataFormats/Common/interface/DetSetVector.h"
typedef edmNew::DetSetVector<MTDTrackingRecHit> MTDTrackingDetSetVector;

#endif
