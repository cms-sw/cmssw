#include "Fireworks/Muons/interface/RPCActiveChamberProxyRhoPhiZ2DBuilder.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveManager.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "TEveStraightLineSet.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "RVersion.h"
#include "TEveGeoNode.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "TColor.h"
#include "TEvePolygonSetProjected.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "Fireworks/Core/interface/FWDisplayEvent.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "Fireworks/Core/src/changeElementAndChildren.h"
#include "TEveCompound.h"

RPCActiveChamberProxyRhoPhiZ2DBuilder::RPCActiveChamberProxyRhoPhiZ2DBuilder()
{
}

RPCActiveChamberProxyRhoPhiZ2DBuilder::~RPCActiveChamberProxyRhoPhiZ2DBuilder()
{
}

void RPCActiveChamberProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem, TEveElementList** product)
{
   build(iItem, product, true);
}

void RPCActiveChamberProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem, TEveElementList** product)
{
   build(iItem, product, false);
}

void RPCActiveChamberProxyRhoPhiZ2DBuilder::build(const FWEventItem* iItem,
					    TEveElementList** product,
					    bool rhoPhiProjection)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"cscSegments",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const RPCRecHitCollection* hits = 0;
   iItem->get(hits);

   if(0 == hits ) {
      std::cout <<"failed to get RPC hits"<<std::endl;
      return;
   }
   unsigned int index = 0;
   const DetIdToMatrix* geom = iItem->getGeom();
   for ( RPCRecHitCollection::id_iterator chamberId = hits->id_begin();
	 chamberId != hits->id_end(); ++chamberId, ++index )
     {
	const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( (*chamberId).rawId() );
	if ( ! matrix ) {
	   std::cout << "ERROR: failed get geometry of RPC reference volume with det id: " <<
	     (*chamberId).rawId() << std::endl;
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

	if ( ! rhoPhiProjection ) {
	   // draw active chamber
	   TEveGeoShape* shape = geom->getShape( (*chamberId).rawId() );
	   if(0!=shape) {
	      shape->SetMainTransparency(75);
	      shape->SetMainColor(iItem->defaultDisplayProperties().color());
	      rpcList->AddElement(shape);
	   }
	} else {
	   if ( (*chamberId).region() == 0 ) { // barrel
	      TEvePointSet* pointSet = new TEvePointSet();
	      pointSet->SetMarkerStyle(2);
	      pointSet->SetMarkerSize(3);
	      pointSet->SetMainColor(iItem->defaultDisplayProperties().color());
	      rpcList->AddElement( pointSet );

	      RPCRecHitCollection::range  range = hits->get(*chamberId);
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

void
RPCActiveChamberProxyRhoPhiZ2DBuilder::modelChanges(const FWModelIds& iIds, TEveElement* iElements)
{
   //NOTE: don't use ids() since they were never filled in in the build* calls

   //for now, only if all items selected will will apply the action
   //if(iIds.size() && iIds.size() == iIds.begin()->item()->size()) {
      applyChangesToAllModels(iElements);
   //}
}

void
RPCActiveChamberProxyRhoPhiZ2DBuilder::applyChangesToAllModels(TEveElement* iElements)
{
   //NOTE: don't use ids() since they may not have been filled in in the build* calls
   //  since this code and FWEventItem do not agree on the # of models made
   //if(ids().size() != 0 ) {
   if(0!=iElements && item() && item()->size()) {
      //make the bad assumption that everything is being changed indentically
      const FWEventItem::ModelInfo info(item()->defaultDisplayProperties(),false);
      changeElementAndChildren(iElements, info);
      iElements->SetRnrSelf(info.displayProperties().isVisible());
      iElements->SetRnrChildren(info.displayProperties().isVisible());
      iElements->ElementChanged();
   }
}

REGISTER_FWRPZ2DDATAPROXYBUILDER(RPCActiveChamberProxyRhoPhiZ2DBuilder,RPCRecHitCollection,"RPCHits");
