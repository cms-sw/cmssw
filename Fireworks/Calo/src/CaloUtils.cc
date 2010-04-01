#include "TEveBox.h"
#include "TEveCompound.h"
#include "Fireworks/Core/interface/FWEventItem.h"

namespace fireworks {

   void addBox(const std::vector<TEveVector> &corners, Color_t color, class TEveElement &oItemHolder)
   {
      const Float_t box[8*3] = { corners[0].fX,  corners[0].fY, corners[0].fZ, 
				 corners[1].fX,  corners[1].fY, corners[1].fZ, 
				 corners[2].fX,  corners[2].fY, corners[2].fZ,
				 corners[3].fX,  corners[3].fY, corners[3].fZ,
				 corners[4].fX,  corners[4].fY, corners[4].fZ,
				 corners[5].fX,  corners[5].fY, corners[5].fZ,
				 corners[6].fX,  corners[6].fY, corners[6].fZ,
				 corners[7].fX,  corners[7].fY, corners[7].fZ};

      TEveBox* eveBox = new TEveBox("Box"); 	 
      eveBox->SetPickable(true);
      eveBox->SetVertices(box);
      eveBox->SetDrawFrame(false);
      eveBox->SetMainColor(color);
	    
      oItemHolder.AddElement(eveBox);
   }

   void addInvertedBox(const std::vector<TEveVector> &corners, Color_t color, class TEveElement &oItemHolder)
   {
      const Float_t box[8*3] = {corners[3].fX,  corners[3].fY, corners[3].fZ, 	 
				corners[2].fX,  corners[2].fY, corners[2].fZ, 	 
				corners[1].fX,  corners[1].fY, corners[1].fZ, 	 
				corners[0].fX,  corners[0].fY, corners[0].fZ, 	 
				corners[7].fX,  corners[7].fY, corners[7].fZ, 	 
				corners[6].fX,  corners[6].fY, corners[6].fZ, 	 
				corners[5].fX,  corners[5].fY, corners[5].fZ, 	 
				corners[4].fX,  corners[4].fY, corners[4].fZ}; 	 

      TEveBox* eveBox = new TEveBox("Box"); 	 
      eveBox->SetPickable(true);
      eveBox->SetVertices(box);
      eveBox->SetDrawFrame(false);
      eveBox->SetMainColor(color);
	    
      oItemHolder.AddElement(eveBox);
   }

   void drawEnergyScaledBox3D(std::vector<TEveVector> &corners, Float_t scale, Color_t color, class TEveElement &oItemHolder)
   {
      TEveVector centre = corners[0] + corners[1] + corners[2] + corners[3] + corners[4] + corners[5] + corners[6] + corners[7];
      centre.Set(centre.fX / 8.0, centre.fY / 8.0, centre.fZ / 8.0);

      // Coordinates for a scaled version of the original box
      for(size_t i = 0; i < 8; ++i)
	corners[i] = centre + (corners[i]-centre)*scale;
      
      addInvertedBox(corners, color, oItemHolder);
   }

   void drawTransverseEnergyScaledBox3D(std::vector<TEveVector> &corners, Float_t scale, Color_t color, class TEveElement &oItemHolder)
   {
      TEveVector centre = corners[0] + corners[1] + corners[2] + corners[3] + corners[4] + corners[5] + corners[6] + corners[7];
      centre.Set(centre.fX / 8.0, centre.fY / 8.0, centre.fZ / 8.0);
      // Calculate transverse energy
      Float_t theta = atan2(sqrt(centre.fX*centre.fX + centre.fY*centre.fY), centre.fZ);
      Float_t et = scale * sin(theta);

      // Coordinates for a scaled version of the original box
      for(size_t i = 0; i < 8; ++i)
	corners[i] = centre + (corners[i]-centre)*et;
      
      addInvertedBox(corners, color, oItemHolder);
   }
  
   void drawEnergyTower3D(std::vector<TEveVector> &corners, Float_t scale, Color_t color, class TEveElement &oItemHolder)
   {
      // Coordinates of a back face scaled 
      for(size_t i = 4; i < 8; ++i)
      {
	TEveVector diff = corners[i]-corners[i-4];
	diff.Normalize();
	corners[i] = corners[i] + (diff * scale);
      }
      addBox(corners, color, oItemHolder);
   }
  
   void drawTransverseEnergyTower3D(std::vector<TEveVector> &corners, Float_t scale, Color_t color, class TEveElement &oItemHolder)
   {
      TEveVector centre = corners[0] + corners[1] + corners[2] + corners[3] + corners[4] + corners[5] + corners[6] + corners[7];
      centre.Set(centre.fX / 8.0, centre.fY / 8.0, centre.fZ / 8.0);
      // Display transverse energy
      Float_t theta = atan2(sqrt(centre.fX*centre.fX + centre.fY*centre.fY), centre.fZ);
      Float_t et = scale * sin(theta);

      // Coordinates of a back face scaled 
      for(size_t i = 4; i < 8; ++i)
      {
	TEveVector diff = corners[i]-corners[i-4];
	diff.Normalize();
	corners[i] = corners[i] + (diff * et);
      }
      addBox(corners, color, oItemHolder);
   }
}
