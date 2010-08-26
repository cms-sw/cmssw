#include "Fireworks/Calo/interface/CaloUtils.h"

#include "TEveBox.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"

#include <algorithm>

namespace fireworks
{
   void invertBox( std::vector<float> &corners )
   {
      std::swap( corners[0], corners[9] );
      std::swap( corners[1], corners[10] );
      std::swap( corners[2], corners[11] );

      std::swap( corners[3], corners[6] );
      std::swap( corners[4], corners[7] );
      std::swap( corners[5], corners[8] );

      std::swap( corners[12], corners[21] );
      std::swap( corners[13], corners[22] );
      std::swap( corners[14], corners[23] );

      std::swap( corners[15], corners[18] );
      std::swap( corners[16], corners[19] );
      std::swap( corners[17], corners[20] );
   }

   void addBox( const std::vector<float> &corners, TEveElement* comp, FWProxyBuilderBase* pb )
   {
      TEveBox* eveBox = new TEveBox( "Box" ); 	 
      eveBox->SetDrawFrame( false );
      eveBox->SetPickable( true );      
      eveBox->SetVertices( &corners[0] );

      pb->setupAddElement( eveBox, comp );
   }

   void drawEnergyScaledBox3D( const float* corners, float scale, TEveElement* comp, FWProxyBuilderBase* pb, bool invert )
   {
      std::vector<float> scaledCorners( 24 );
      std::vector<float> centre( 3, 0 );

      for( unsigned int i = 0; i < 24; i += 3 )
      {	 
	centre[0] += corners[i];
	centre[1] += corners[i + 1];
	centre[2] += corners[i + 2];
      }

      for( unsigned int i = 0; i < 3; ++i )
	centre[i] *= 1.0f / 8.0f;

       // Coordinates for a scaled version of the original box
      for( unsigned int i = 0; i < 24; i += 3 )
      {	
	scaledCorners[i] = centre[0] + ( corners[i] - centre[0] ) * scale;
	scaledCorners[i + 1] = centre[1] + ( corners[i + 1] - centre[1] ) * scale;
	scaledCorners[i + 2] = centre[2] + ( corners[i + 2] - centre[2] ) * scale;
      }
      
      if( invert )
	 invertBox( scaledCorners );

      addBox( scaledCorners, comp, pb );
   }

   void drawEnergyTower3D( const float* corners, float scale, TEveElement* comp, FWProxyBuilderBase* pb, bool reflect )
   {
     std::vector<float> scaledCorners( 24 );
     for( int i = 0; i < 24; ++i )
        scaledCorners[i] = corners[i];
      // Coordinates of a front face scaled 
      if( reflect )
      {
	 // We know, that an ES rechit geometry in -Z needs correction. 
	 // The back face is actually its front face.
	 for( unsigned int i = 0; i < 12; i += 3 )
	 {
	    TEveVector diff( corners[i] - corners[i + 12], corners[i + 1] - corners[i + 13], corners[i + 2] - corners[i + 14] );
	    diff.Normalize();
	    diff *= scale;
	    
	    scaledCorners[i] = corners[i] + diff.fX;
	    scaledCorners[i + 1] = corners[i + 1] + diff.fY;
	    scaledCorners[i + 2] = corners[i + 2] + diff.fZ;
	 }
      } 
      else
      {
	 for( unsigned int i = 0; i < 12; i += 3 )
	 {
	    TEveVector diff( corners[i + 12] - corners[i], corners[i + 13] - corners[i + 1], corners[i + 14] - corners[i + 2] );
	    diff.Normalize();
	    diff *= scale;
	    
	    scaledCorners[i] = corners[i + 12];
	    scaledCorners[i + 1] = corners[i + 13];
	    scaledCorners[i + 2] = corners[i + 14];
	    
	    scaledCorners[i + 12] = corners[i + 12] + diff.fX;
	    scaledCorners[i + 13] = corners[i + 13] + diff.fY;
	    scaledCorners[i + 14] = corners[i + 14] + diff.fZ;
	 }
      }
      addBox( scaledCorners, comp, pb );
   }
  
   void drawEtTower3D( const float* corners, float scale, TEveElement* comp, FWProxyBuilderBase* pb, bool reflect )
   {
     std::vector<float> scaledCorners( 24 );
     for( int i = 0; i < 24; ++i )
        scaledCorners[i] = corners[i];
      // Coordinates of a front face scaled 
      if( reflect )
      {
	 // We know, that an ES rechit geometry in -Z needs correction. 
	 // The back face is actually its front face.
	 for( unsigned int i = 0; i < 12; i += 3 )
	 {
	    TEveVector diff( corners[i] - corners[i + 12], corners[i + 1] - corners[i + 13], corners[i + 2] - corners[i + 14] );
	    diff.Normalize();
	    diff *= ( scale * sin( diff.Theta()));
	    
	    scaledCorners[i] = corners[i] + diff.fX;
	    scaledCorners[i + 1] = corners[i + 1] + diff.fY;
	    scaledCorners[i + 2] = corners[i + 2] + diff.fZ;
	 }
      } 
      else
      {
	 for( unsigned int i = 0; i < 12; i += 3 )
	 {
	    TEveVector diff( corners[i + 12] - corners[i], corners[i + 13] - corners[i + 1], corners[i + 14] - corners[i + 2] );
	    diff.Normalize();
	    diff *= ( scale * sin( diff.Theta()));
	    
	    scaledCorners[i] = corners[i + 12];
	    scaledCorners[i + 1] = corners[i + 13];
	    scaledCorners[i + 2] = corners[i + 14];
	    
	    scaledCorners[i + 12] = corners[i + 12] + diff.fX;
	    scaledCorners[i + 13] = corners[i + 13] + diff.fY;
	    scaledCorners[i + 14] = corners[i + 14] + diff.fZ;
	 }
      }
      addBox( scaledCorners, comp, pb );
   }
}
