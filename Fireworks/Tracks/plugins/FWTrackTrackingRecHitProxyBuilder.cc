#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"

class FWTrackTrackingRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track> {
public:
  FWTrackTrackingRecHitProxyBuilder(void) {}
  ~FWTrackTrackingRecHitProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  using FWSimpleProxyBuilderTemplate<reco::Track>::build;
  void build(const reco::Track& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) override;

  FWTrackTrackingRecHitProxyBuilder(const FWTrackTrackingRecHitProxyBuilder&) = delete;  // stop default
  const FWTrackTrackingRecHitProxyBuilder& operator=(const FWTrackTrackingRecHitProxyBuilder&) = delete;  // stop default
};

void FWTrackTrackingRecHitProxyBuilder::build(const reco::Track& iData,
                                              unsigned int iIndex,
                                              TEveElement& oItemHolder,
                                              const FWViewContext*) {
  const FWGeometry* geom = item()->getGeom();

  for (trackingRecHit_iterator it = iData.recHitsBegin(), itEnd = iData.recHitsEnd(); it != itEnd; ++it) {
    TEvePointSet* pointSet = new TEvePointSet;
    setupAddElement(pointSet, &oItemHolder);

    auto rechitRef = *it;
    const TrackingRecHit* rechit = &(*rechitRef);

    if (rechit->isValid()) {
      unsigned int rawid = rechit->geographicalId().rawId();

      if (!geom->contains(rawid)) {
        fwLog(fwlog::kError) << "failed get geometry for detid: " << rawid << std::endl;
      }

      LocalPoint pos(0.0, 0.0, 0.0);
      if (const SiStripRecHit2D* hit = dynamic_cast<const SiStripRecHit2D*>(rechit)) {
        if (hit->hasPositionAndError()) {
          pos = rechit->localPosition();
        }
      } else if (const SiStripRecHit1D* hit = dynamic_cast<const SiStripRecHit1D*>(rechit)) {
        if (hit->hasPositionAndError()) {
          pos = rechit->localPosition();
        }
      }

      float localPos[3] = {pos.x(), pos.y(), pos.z()};
      float globalPos[3];
      geom->localToGlobal(rawid, localPos, globalPos);
      pointSet->SetNextPoint(globalPos[0], globalPos[1], globalPos[2]);
    }
  }
}

REGISTER_FWPROXYBUILDER(FWTrackTrackingRecHitProxyBuilder,
                        reco::Track,
                        "Track Tracking RecHits",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
