// -*- C++ -*-
// $Id: FWTracksRecHitsProxyBuilder.cc,v 1.1 2009/01/16 10:37:00 Tom Danielson
//

// user include files
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Tracks/interface/TracksRecHitsUtil.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

class FWTracksRecHitsProxyBuilder : public FWProxyBuilderBase
{
public:
   FWTracksRecHitsProxyBuilder() {
   }
   virtual ~FWTracksRecHitsProxyBuilder() {
   }
  
   REGISTER_PROXYBUILDER_METHODS();
private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);

   FWTracksRecHitsProxyBuilder(const FWTracksRecHitsProxyBuilder&);    // stop default
   const FWTracksRecHitsProxyBuilder& operator=(const FWTracksRecHitsProxyBuilder&);    // stop default
};

void
FWTracksRecHitsProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TracksRecHitsUtil::buildTracksRecHits(iItem, product, true, false);
}

class FWTracksModulesProxyBuilder : public FWProxyBuilderBase
{
public:
   FWTracksModulesProxyBuilder() {
   }
   virtual ~FWTracksModulesProxyBuilder() {
   }

   REGISTER_PROXYBUILDER_METHODS();
private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);

   FWTracksModulesProxyBuilder(const FWTracksModulesProxyBuilder&);    // stop default
   const FWTracksModulesProxyBuilder& operator=(const FWTracksModulesProxyBuilder&);    // stop default
};

void
FWTracksModulesProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TracksRecHitsUtil::buildTracksRecHits(iItem, product, false, true);
}

REGISTER_FWPROXYBUILDER(FWTracksRecHitsProxyBuilder,reco::TrackCollection,"TrackHits", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWTracksModulesProxyBuilder,reco::TrackCollection,"TrackDets", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
