// -*- C++ -*-
//
// Package:     Muons
// Class  :     RPCActiveChamberProxyRhoPhiZ2DBuilder
//
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: FWRPCRecHitProxyBuilder.cc,v 1.4 2010/04/08 13:09:33 yana Exp $
//

#include "TEveStraightLineSet.h"
#include "TEveManager.h"
#include "TEveGeoNode.h"
#include "TEveCompound.h"
#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

class FWRPCRecHitProxyBuilder : public FWProxyBuilderBase
{
public:
   FWRPCRecHitProxyBuilder() {}
   virtual ~FWRPCRecHitProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem, TEveElementList** product);

   FWRPCRecHitProxyBuilder(const FWRPCRecHitProxyBuilder&);    // stop default

   const FWRPCRecHitProxyBuilder& operator=(const FWRPCRecHitProxyBuilder&);    // stop default
};

void
FWRPCRecHitProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   bool rhoPhiProjection = true;
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList( iItem->name().c_str(), "rpcRecHits", true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const RPCRecHitCollection* hits = 0;
   iItem->get(hits);

   if(0 == hits ) {
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
      TEveCompound* rpcList = new TEveCompound(title, title);
      rpcList->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(rpcList,boost::mem_fn(&TEveCompound::CloseCompound));
      gEve->AddElement( rpcList, tList );
      rpcList->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
      rpcList->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );

      if ( !rhoPhiProjection ) {
         // draw active chamber
         TEveGeoShape* shape = geom->getShape( (*chamberId).rawId() );
         if(0!=shape) {
            shape->SetMainTransparency(75);
            shape->SetMainColor(iItem->defaultDisplayProperties().color());
            rpcList->AddElement(shape);
         }
      } else {
         if ( (*chamberId).region() == 0 ) {   // barrel
            TEvePointSet* pointSet = new TEvePointSet();
            pointSet->SetMarkerStyle(2);
            pointSet->SetMarkerSize(3);
            pointSet->SetMainColor(iItem->defaultDisplayProperties().color());
            rpcList->AddElement( pointSet );

            RPCRecHitCollection::range range = hits->get(*chamberId);
            for (RPCRecHitCollection::const_iterator hit = range.first;
                 hit!=range.second; ++hit)
            {
               Double_t localPoint[3];
               Double_t globalPoint[3];

               localPoint[0] = hit->localPosition().x();
               localPoint[1] = hit->localPosition().y();
               localPoint[2] = hit->localPosition().z();

               if ( (*chamberId).layer() == 1 && (*chamberId).station() < 3 ) localPoint[0] = -localPoint[0];
               matrix->LocalToMaster( localPoint, globalPoint );
               // printf("RPC id: %d \t(%0.2f,%0.2f,%0.2f)->(%0.2f,%0.2f,%0.2f)\n",
               // (*chamberId).rawId(), localPoint[0], localPoint[1], localPoint[2], globalPoint[0], globalPoint[1], globalPoint[2]);
               pointSet->SetNextPoint( globalPoint[0], globalPoint[1], globalPoint[2] );
            }
         }
      }
   }
}

REGISTER_FWPROXYBUILDER( FWRPCRecHitProxyBuilder, RPCRecHitCollection, "RPCHits", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
