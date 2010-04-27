#include "Fireworks/Calo/interface/CaloUtils.h"

#include "TEveBox.h"
#include "TEveCompound.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"

namespace fireworks {

void addBox( const std::vector<TEveVector> &corners, TEveElement* comp, FWProxyBuilderBase* pb)
{
   const Float_t box[8*3] = { corners[0].fX,  corners[0].fY, corners[0].fZ, 
                              corners[1].fX,  corners[1].fY, corners[1].fZ, 
                              corners[2].fX,  corners[2].fY, corners[2].fZ,
                              corners[3].fX,  corners[3].fY, corners[3].fZ,
                              corners[4].fX,  corners[4].fY, corners[4].fZ,
                              corners[5].fX,  corners[5].fY, corners[5].fZ,
                              corners[6].fX,  corners[6].fY, corners[6].fZ,
                              corners[7].fX,  corners[7].fY, corners[7].fZ};

   TEveBox* eveBox = new TEveBox( "Box" ); 	 
   eveBox->SetPickable( true );
   eveBox->SetVertices( box );
   eveBox->SetDrawFrame( false );
	    
   pb->setupAddElement( eveBox, comp );
}

void addInvertedBox( const std::vector<TEveVector> &corners, TEveElement* comp, FWProxyBuilderBase* pb)
{
   const Float_t box[8*3] = {corners[3].fX,  corners[3].fY, corners[3].fZ, 	 
                             corners[2].fX,  corners[2].fY, corners[2].fZ, 	 
                             corners[1].fX,  corners[1].fY, corners[1].fZ, 	 
                             corners[0].fX,  corners[0].fY, corners[0].fZ, 	 
                             corners[7].fX,  corners[7].fY, corners[7].fZ, 	 
                             corners[6].fX,  corners[6].fY, corners[6].fZ, 	 
                             corners[5].fX,  corners[5].fY, corners[5].fZ, 	 
                             corners[4].fX,  corners[4].fY, corners[4].fZ}; 	 

   TEveBox* eveBox = new TEveBox( "Box" ); 	 
   eveBox->SetPickable( true );
   eveBox->SetVertices( box );
   eveBox->SetDrawFrame( false );
	    
   pb->setupAddElement( eveBox, comp, pb  );
}

void drawEnergyScaledBox3D( std::vector<TEveVector> &corners, float scale, TEveElement* comp, FWProxyBuilderBase* pb)
{
   TEveVector centre = corners[0] + corners[1] + corners[2] + corners[3] + corners[4] + corners[5] + corners[6] + corners[7];
   centre.Set( centre.fX / 8.0, centre.fY / 8.0, centre.fZ / 8.0 );

   // Coordinates for a scaled version of the original box
   for( size_t i = 0; i < 8; ++i )
      corners[i] = centre + (corners[i]-centre)*scale;
      
   addInvertedBox( corners, comp, pb );
}

void drawEnergyTower3D( std::vector<TEveVector> &corners, float scale, TEveElement* comp, FWProxyBuilderBase* pb)
{
   // Coordinates of a back face scaled 
   for( size_t i = 4; i < 8; ++i )
   {
      TEveVector diff = corners[i]-corners[i-4];
      diff.Normalize();
      corners[i] = corners[i] + (diff * scale);
   }
   addBox( corners, comp, pb );
}
}
