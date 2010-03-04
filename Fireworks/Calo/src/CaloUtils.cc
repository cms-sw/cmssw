#include "TEveBoxSet.h"
#include "TEveCompound.h"
#include "Fireworks/Core/interface/FWEventItem.h"

namespace fireworks {

   void drawCaloHit3D(std::vector<TEveVector> &corners, const FWEventItem* iItem, class TEveElement &oItemHolder, Float_t scaleFraction)
   {
      TEveVector centre = corners[0] + corners[1] + corners[2] + corners[3] + corners[4] + corners[5] + corners[6] + corners[7];
      centre.Set(centre.fX / 8.0, centre.fY / 8.0, centre.fZ / 8.0);

      // Coordinates for a scaled version of the original box
      for(size_t i = 0; i < 8; ++i)
	corners[i] = centre + (corners[i]-centre)*scaleFraction;
      
      const Float_t box[8*3] = {corners[3].fX,  corners[3].fY, corners[3].fZ, 	 
				corners[2].fX,  corners[2].fY, corners[2].fZ, 	 
				corners[1].fX,  corners[1].fY, corners[1].fZ, 	 
				corners[0].fX,  corners[0].fY, corners[0].fZ, 	 
				corners[7].fX,  corners[7].fY, corners[7].fZ, 	 
				corners[6].fX,  corners[6].fY, corners[6].fZ, 	 
				corners[5].fX,  corners[5].fY, corners[5].fZ, 	 
				corners[4].fX,  corners[4].fY, corners[4].fZ}; 	 

      // FIXME: We do not need to make a box set per hit
      // but rather one box set per collection...
      TEveBoxSet* recHit = new TEveBoxSet("Rec Hit"); 	 
      recHit->Reset(TEveBoxSet::kBT_FreeBox, kTRUE, 1);
      recHit->SetPickable(true);
      recHit->AddBox(box);
      recHit->DigitColor(iItem->defaultDisplayProperties().color());
	    
      oItemHolder.AddElement(recHit);
   }
  
   void drawEcalHit3D(std::vector<TEveVector> &corners, const FWEventItem* iItem, class TEveElement &oItemHolder, Float_t scale)
   {
      // Coordinates of a back face scaled 
      for(size_t i = 4; i < 8; ++i)
      {
	TEveVector diff = corners[i]-corners[i-4];
	diff.Normalize();
	corners[i] = corners[i] + (diff * scale);
      }
      
      const Float_t box[8*3] = { corners[0].fX,  corners[0].fY, corners[0].fZ, 
				 corners[1].fX,  corners[1].fY, corners[1].fZ, 
				 corners[2].fX,  corners[2].fY, corners[2].fZ,
				 corners[3].fX,  corners[3].fY, corners[3].fZ,
				 corners[4].fX,  corners[4].fY, corners[4].fZ,
				 corners[5].fX,  corners[5].fY, corners[5].fZ,
				 corners[6].fX,  corners[6].fY, corners[6].fZ,
				 corners[7].fX,  corners[7].fY, corners[7].fZ};

      // FIXME: We do not need to make a box set per hit
      // but rather one box set per collection...
      TEveBoxSet* recHit = new TEveBoxSet("Rec Hit"); 	 
      recHit->Reset(TEveBoxSet::kBT_FreeBox, kTRUE, 1); 	 
      recHit->SetPickable(true);
      recHit->AddBox(box);
      recHit->DigitColor(iItem->defaultDisplayProperties().color());
	    
      oItemHolder.AddElement(recHit);
   }
}
