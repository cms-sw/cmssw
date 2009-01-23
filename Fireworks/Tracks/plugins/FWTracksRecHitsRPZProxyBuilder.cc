// -*- C++ -*-
// $Id: FWTracksRecHitsRPZProxyBuilder.cc,v 1.1 2009/01/16 10:37:00 Tom Danielson
//

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Tracks/plugins/TracksRecHitsUtil.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/src/changeElementAndChildren.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

class FWTracksRecHitsRPZProxyBuilder : public FWRPZDataProxyBuilder
{
public:
   FWTracksRecHitsRPZProxyBuilder() {
   }
   virtual ~FWTracksRecHitsRPZProxyBuilder() {
   }

   // ---------- const member functions ---------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();
private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);

   FWTracksRecHitsRPZProxyBuilder(const FWTracksRecHitsRPZProxyBuilder&);    // stop default

   const FWTracksRecHitsRPZProxyBuilder& operator=(const FWTracksRecHitsRPZProxyBuilder&);    // stop default

   void modelChanges(const FWModelIds& iIds, TEveElement* iElements);
   void applyChangesToAllModels(TEveElement* iElements);
   // ---------- member data --------------------------------

};

void FWTracksRecHitsRPZProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TracksRecHitsUtil::buildTracksRecHits(iItem, product);
}

void
FWTracksRecHitsRPZProxyBuilder::modelChanges(const FWModelIds& iIds, TEveElement* iElements)
{
   applyChangesToAllModels(iElements);
}

void
FWTracksRecHitsRPZProxyBuilder::applyChangesToAllModels(TEveElement* iElements)
{
   if(0!=iElements && item() && item()->size()) {
      //make the bad assumption that everything is being changed indentically
      const FWEventItem::ModelInfo info(item()->defaultDisplayProperties(),false);
      changeElementAndChildren(iElements, info);
      iElements->SetRnrSelf(info.displayProperties().isVisible());
      iElements->SetRnrChildren(info.displayProperties().isVisible());
      iElements->ElementChanged();
   }
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWTracksRecHitsRPZProxyBuilder,reco::TrackCollection,"TrackHits");

