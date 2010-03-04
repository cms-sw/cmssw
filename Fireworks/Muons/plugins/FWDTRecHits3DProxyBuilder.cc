#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveCompound.h"
#include "TEveStraightLineSet.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

using namespace DTEnums;

class FWDTRecHits3DProxyBuilder : public FW3DDataProxyBuilder
{
public:
   FWDTRecHits3DProxyBuilder(void) 
    {}
  
   virtual ~FWDTRecHits3DProxyBuilder(void) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product);

   // Disable default copy constructor
   FWDTRecHits3DProxyBuilder(const FWDTRecHits3DProxyBuilder&); 
   // Disable default assignment operator
   const FWDTRecHits3DProxyBuilder& operator=(const FWDTRecHits3DProxyBuilder&);
};

void
FWDTRecHits3DProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(), "dtRechits", true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const DTRecHitCollection* collection = 0;
   iItem->get(collection);

   if(0 == collection)
   {
      return;
   }
   TEveCompound* compund = new TEveCompound("dt compound", "dtRechits");
   compund->OpenCompound();
   TEveStraightLineSet* rechitSet = new TEveStraightLineSet("DT RecHit Collection");
   rechitSet->SetMainColor(iItem->defaultDisplayProperties().color());
   rechitSet->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
   rechitSet->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
   compund->AddElement(rechitSet);   

   for(DTRecHitCollection::id_iterator chId = collection->id_begin(), chIdEnd = collection->id_end();
       chId != chIdEnd; ++chId)
   {
      const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix((*chId).chamberId());
      if(!matrix) {
         std::cout << "ERROR: failed get geometry of DT chamber with det id: " <<
         (*chId).chamberId() << std::endl;
         continue;
      }

      Double_t localCenterPoint[3] = {0.0, 0.0, 0.0};
      Double_t globalCenterPoint[3];
      
      DTRecHitCollection::range range = collection->get(*chId);
      for(DTRecHitCollection::const_iterator it = range.first;
	  it != range.second; ++it)
      {
	 Double_t localPos[3] = {(*it).localPosition().x(), (*it).localPosition().y(), (*it).localPosition().z()};
	 Double_t globalPos[3];
	 
	 const DTRecHit1D* lrechit = (*it).componentRecHit(Left);
         const DTRecHit1D* rrechit = (*it).componentRecHit(Right);

	 Double_t lLocalPos[3] = {lrechit->localPosition().x(), lrechit->localPosition().y(), lrechit->localPosition().z()};
	 Double_t rLocalPos[3] = {rrechit->localPosition().x(), rrechit->localPosition().y(), rrechit->localPosition().z()};

	 Double_t lGlobalPoint[3];
         Double_t rGlobalPoint[3];

         matrix->LocalToMaster(lLocalPos, lGlobalPoint);
         matrix->LocalToMaster(rLocalPos, rGlobalPoint);
         matrix->LocalToMaster(localCenterPoint, globalCenterPoint);
         matrix->LocalToMaster(localPos, globalPos);
	 
	 rechitSet->AddLine(lGlobalPoint[0], lGlobalPoint[1], lGlobalPoint[2], rGlobalPoint[0], rGlobalPoint[1], rGlobalPoint[2]);
	 rechitSet->AddLine(globalCenterPoint[0], globalCenterPoint[1], globalCenterPoint[2], rGlobalPoint[0], rGlobalPoint[1], rGlobalPoint[2]);
	 rechitSet->AddLine(lGlobalPoint[0], lGlobalPoint[1], lGlobalPoint[2], globalCenterPoint[0], globalCenterPoint[1], globalCenterPoint[2]);

	 rechitSet->AddLine(globalPos[0], globalPos[1], globalPos[2], globalCenterPoint[0], globalCenterPoint[1], globalCenterPoint[2]);
      }
   }
   tList->AddElement(compund);
}

REGISTER_FW3DDATAPROXYBUILDER(FWDTRecHits3DProxyBuilder, DTRecHitCollection, "DT Rec Hits");


