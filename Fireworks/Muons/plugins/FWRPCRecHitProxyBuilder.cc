// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWRPCRecHitProxyBuilder
//
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: FWRPCRecHitProxyBuilder.cc,v 1.2 2010/04/16 13:11:30 yana Exp $
//

#include "TEveStraightLineSet.h"
#include "TEveManager.h"
#include "TEveGeoNode.h"
#include "TEveCompound.h"
#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

class FWRPCRecHitProxyBuilder : public FWProxyBuilderBase
{
public:
   FWRPCRecHitProxyBuilder() {}
   virtual ~FWRPCRecHitProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWRPCRecHitProxyBuilder(const FWRPCRecHitProxyBuilder&);    // stop default
   const FWRPCRecHitProxyBuilder& operator=(const FWRPCRecHitProxyBuilder&);    // stop default
  
   virtual void build(const FWEventItem* iItem, TEveElementList* product);
};

void
FWRPCRecHitProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product)
{
   const RPCRecHitCollection* hits = 0;
   iItem->get(hits);

   if( 0 == hits ) {
      return;
   }
   
   unsigned int index = 0;
   const DetIdToMatrix* geom = iItem->getGeom();
   for( RPCRecHitCollection::id_iterator chamberId = hits->id_begin(), chamberIdEnd = hits->id_end();
	chamberId != chamberIdEnd; ++chamberId, ++index )
   {
      const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( (*chamberId).rawId() );
      if ( !matrix ) {
         std::cout << "ERROR: failed get geometry of RPC reference volume with det id: "
		   << (*chamberId).rawId() << std::endl;
         continue;
      }

      const unsigned int nBuffer = 1024;
      char title[nBuffer];
      snprintf(title, nBuffer,"RPC module %d", (*chamberId).rawId());

      TEveGeoShape* shape = geom->getShape( (*chamberId).rawId() );

      RPCRecHitCollection::range range = hits->get(*chamberId);
      for( RPCRecHitCollection::const_iterator hit = range.first;
	   hit != range.second; ++hit)
      {
         TEveCompound* compund = new TEveCompound("rpc compound", title );
         compund->OpenCompound();
         product->AddElement(compund);

	 Double_t localPoint[3];
	 Double_t globalPoint[3];

	 localPoint[0] = hit->localPosition().x();
	 localPoint[1] = hit->localPosition().y();
	 localPoint[2] = hit->localPosition().z();
	 
	 TEvePointSet* pointSet = new TEvePointSet;
	 pointSet->SetMarkerStyle( 2 );
	 pointSet->SetMarkerSize( 3 );
         pointSet->SetMainColor( iItem->defaultDisplayProperties().color() );
         pointSet->SetRnrSelf( iItem->defaultDisplayProperties().isVisible() );
         pointSet->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );
         compund->AddElement( pointSet );
         if( 0 != shape ) {
            shape->SetMainTransparency( 75 );
            shape->SetMainColor( iItem->defaultDisplayProperties().color() );
            pointSet->AddElement( shape );
         }

	 if( (*chamberId).layer() == 1 && (*chamberId).station() < 3 )
	    localPoint[0] = -localPoint[0];
	 matrix->LocalToMaster( localPoint, globalPoint );
	 pointSet->SetNextPoint( globalPoint[0], globalPoint[1], globalPoint[2] );
      }
   }
}

REGISTER_FWPROXYBUILDER( FWRPCRecHitProxyBuilder, RPCRecHitCollection, "RPC RecHits", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
