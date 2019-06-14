// -*- C++ -*-
// $Id: FWTracksRecHitsProxyBuilder.cc,v 1.1 2009/01/16 10:37:00 Tom Danielson
//

// user include files
#include "TEveGeoShape.h"
#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "Fireworks/Core/interface/fwLog.h"

class FWTracksRecHitsProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track> {
public:
  FWTracksRecHitsProxyBuilder(void) {}
  ~FWTracksRecHitsProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

  static bool representsSubPart(void);

private:
  using FWSimpleProxyBuilderTemplate<reco::Track>::build;
  void build(const reco::Track& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) override;

  FWTracksRecHitsProxyBuilder(const FWTracksRecHitsProxyBuilder&) = delete;                   // stop default
  const FWTracksRecHitsProxyBuilder& operator=(const FWTracksRecHitsProxyBuilder&) = delete;  // stop default
};

void FWTracksRecHitsProxyBuilder::build(const reco::Track& track,
                                        unsigned int iIndex,
                                        TEveElement& oItemHolder,
                                        const FWViewContext*) {
  if (track.extra().isAvailable()) {
    std::vector<TVector3> points;
    const FWEventItem& iItem = *item();
    fireworks::pushPixelHits(points, iItem, track);

    TEvePointSet* pointSet = new TEvePointSet();
    for (std::vector<TVector3>::const_iterator it = points.begin(), itEnd = points.end(); it != itEnd; ++it) {
      pointSet->SetNextPoint(it->x(), it->y(), it->z());
    }
    setupAddElement(pointSet, &oItemHolder);

    fireworks::addSiStripClusters(item(), track, &oItemHolder, false, true);
  }
}

bool FWTracksRecHitsProxyBuilder::representsSubPart(void) { return true; }

REGISTER_FWPROXYBUILDER(FWTracksRecHitsProxyBuilder,
                        reco::Track,
                        "TrackHits",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
