// -*- C++ -*-
// $Id: FWTracksRecHits3DProxyBuilder.cc,v 1.1 2009/01/16 10:37:00 Tom Danielson
//

// user include files
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "Fireworks/Tracks/plugins/TracksRecHitsUtil.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

class FWTracksRecHits3DProxyBuilder : public FW3DDataProxyBuilder
{
public:
   FWTracksRecHits3DProxyBuilder() {
   }
   virtual ~FWTracksRecHits3DProxyBuilder() {
   }

   REGISTER_PROXYBUILDER_METHODS();
private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);

   FWTracksRecHits3DProxyBuilder(const FWTracksRecHits3DProxyBuilder&);    // stop default
   const FWTracksRecHits3DProxyBuilder& operator=(const FWTracksRecHits3DProxyBuilder&);    // stop default

};

void FWTracksRecHits3DProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TracksRecHitsUtil::buildTracksRecHits(iItem, product);
}

REGISTER_FW3DDATAPROXYBUILDER(FWTracksRecHits3DProxyBuilder,reco::TrackCollection,"TrackHits");
