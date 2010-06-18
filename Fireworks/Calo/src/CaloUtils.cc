#include "Fireworks/Calo/interface/CaloUtils.h"

#include "TEveBox.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"

#include <algorithm>

namespace
{
   void invertBox( std::vector<TEveVector> &corners )
   {
      std::swap( corners[0], corners[3] );
      std::swap( corners[1], corners[2] );
      std::swap( corners[4], corners[7] );
      std::swap( corners[5], corners[6] );
   }
}

namespace fireworks
{

   void addBox( const std::vector<TEveVector> &corners, TEveElement* comp, FWProxyBuilderBase* pb )
   {
      TEveBox* eveBox = new TEveBox( "Box" ); 	 
      eveBox->SetPickable( true );
      for( int i = 0; i < 8; ++i )
	 eveBox->SetVertex( i, corners[i] );
      eveBox->SetDrawFrame( false );

      pb->setupAddElement( eveBox, comp );
   }

   void drawEnergyScaledBox3D( std::vector<TEveVector> &corners, float scale, TEveElement* comp, FWProxyBuilderBase* pb, bool invert )
   {
      TEveVector centre = corners[0] + corners[1] + corners[2] + corners[3] + corners[4] + corners[5] + corners[6] + corners[7];
      centre *= 1.0f / 8.0f;

      // Coordinates for a scaled version of the original box
      for( size_t i = 0; i < 8; ++i )
	 corners[i] = centre + ( corners[i] - centre ) * scale;

      if( invert )
	 invertBox( corners );

      addBox( corners, comp, pb );
   }

   void drawEnergyTower3D( std::vector<TEveVector> &corners, float scale, TEveElement* comp, FWProxyBuilderBase* pb, bool reflect )
   {
      // Coordinates of a front face scaled 
      if( reflect )
      {
	 // We know, that an ES rechit geometry in -Z needs correction. 
	 // The back face is actually its front face.
	 for( size_t i = 0; i < 4; ++i )
	 {
	    TEveVector diff = corners[i] - corners[i+4];
	    diff.Normalize();
	    corners[i] = corners[i] + (diff * scale);
	 }
      } 
      else
      {
	 for( size_t i = 4; i < 8; ++i )
	 {
	    TEveVector diff = corners[i] - corners[i-4];
	    diff.Normalize();
	    corners[i-4] = corners[i];
	    corners[i] = corners[i] + (diff * scale);
	 }
      }
      addBox( corners, comp, pb );
   }
}
