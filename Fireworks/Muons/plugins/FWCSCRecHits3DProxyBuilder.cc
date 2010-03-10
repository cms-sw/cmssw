#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveCompound.h"
#include "TEveStraightLineSet.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

class FWCSCRecHits3DProxyBuilder : public FW3DDataProxyBuilder
{
public:
   FWCSCRecHits3DProxyBuilder(void) 
    {}
  
   virtual ~FWCSCRecHits3DProxyBuilder(void)
    {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product);

   // Disable default copy constructor
   FWCSCRecHits3DProxyBuilder(const FWCSCRecHits3DProxyBuilder&);
   // Disable default assignment operator
   const FWCSCRecHits3DProxyBuilder& operator=(const FWCSCRecHits3DProxyBuilder&);
};

void
FWCSCRecHits3DProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(), "cscRechits", true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const CSCRecHit2DCollection* collection = 0;
   iItem->get(collection);

   if(0 == collection)
   {
      return;
   }

   unsigned int index = 0;
   for(CSCRecHit2DCollection::id_iterator chId = collection->id_begin(), chIdEnd = collection->id_end();
       chId != chIdEnd; ++chId, ++index)
   {
      const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix(*chId);
      if(!matrix) {
         std::cout << "ERROR: failed get geometry of CSC layer with det id: " <<
	   (*chId) << std::endl;
         continue;
      }

      std::stringstream s;
      s << "chamber" << index;

      CSCRecHit2DCollection::range range = collection->get(*chId);
      for(CSCRecHit2DCollection::const_iterator it = range.first;
	  it != range.second; ++it)
      {
	 TEveCompound* compund = new TEveCompound("csc compound", "cscRechits");
	 compund->OpenCompound();
	 tList->AddElement(compund);
  
	 TEveStraightLineSet* rechitSet = new TEveStraightLineSet(s.str().c_str());
	 rechitSet->SetMainColor(iItem->defaultDisplayProperties().color());
	 rechitSet->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
	 rechitSet->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
	 compund->AddElement(rechitSet);
	 
	 Float_t x = it->localPosition().x();
	 Float_t y = it->localPosition().y();
	 Float_t z = 0.0;
	 Float_t dx = sqrt(it->localPositionError().xx());
	 Float_t dy = sqrt(it->localPositionError().yy());

         Double_t localU1Point[3] = {x - dx, y, z};
         Double_t localU2Point[3] = {x + dx, y, z};
         Double_t localV1Point[3] = {x, y - dy, z};
         Double_t localV2Point[3] = {x, y + dy, z};

         Double_t globalU1Point[3];
         Double_t globalU2Point[3];
         Double_t globalV1Point[3];
         Double_t globalV2Point[3];

         matrix->LocalToMaster(localU1Point, globalU1Point);
         matrix->LocalToMaster(localU2Point, globalU2Point);
         matrix->LocalToMaster(localV1Point, globalV1Point);
         matrix->LocalToMaster(localV2Point, globalV2Point);
	 rechitSet->AddLine(globalU1Point[0], globalU1Point[1], globalU1Point[2], globalU2Point[0], globalU2Point[1], globalU2Point[2]);
	 rechitSet->AddLine(globalV1Point[0], globalV1Point[1], globalV1Point[2], globalV2Point[0], globalV2Point[1], globalV2Point[2]);
      }
   }
}

REGISTER_FW3DDATAPROXYBUILDER(FWCSCRecHits3DProxyBuilder, CSCRecHit2DCollection, "CSC Rec Hits");


