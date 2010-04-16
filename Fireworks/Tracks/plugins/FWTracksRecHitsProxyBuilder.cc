// -*- C++ -*-
// $Id: FWTracksRecHitsProxyBuilder.cc,v 1.1 2009/01/16 10:37:00 Tom Danielson
//

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/TrackReco/interface/Track.h"

class FWTracksRecHitsProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track>
{
public:
   FWTracksRecHitsProxyBuilder() {}
   virtual ~FWTracksRecHitsProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();
  
private:
   void build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   FWTracksRecHitsProxyBuilder(const FWTracksRecHitsProxyBuilder&);    // stop default
   const FWTracksRecHitsProxyBuilder& operator=(const FWTracksRecHitsProxyBuilder&);    // stop default
};

void
FWTracksRecHitsProxyBuilder::build(const reco::Track& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   fireworks::addHits(iData, item(), &oItemHolder, false);
}

class FWTracksModulesProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track>
{
public:
   FWTracksModulesProxyBuilder() {}
   virtual ~FWTracksModulesProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();
  
private:
   void build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   FWTracksModulesProxyBuilder(const FWTracksModulesProxyBuilder&);    // stop default
   const FWTracksModulesProxyBuilder& operator=(const FWTracksModulesProxyBuilder&);    // stop default
};

void
FWTracksModulesProxyBuilder::build(const reco::Track& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   fireworks::addModules(iData, item(), &oItemHolder, false);
}

REGISTER_FWPROXYBUILDER(FWTracksRecHitsProxyBuilder, reco::Track, "TrackHits", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
REGISTER_FWPROXYBUILDER(FWTracksModulesProxyBuilder, reco::Track, "TrackDets", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
