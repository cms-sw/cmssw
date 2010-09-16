// #include "Fireworks/Calo/interface/CaloUtils.h"

// #include "TEveBox.h"
// #include "TEveScalableStraightLineSet.h"
// #include "TEveStraightLineSet.h"

// #include "Fireworks/Core/interface/FWProxyBuilderBase.h"
// #include "Fireworks/Core/interface/Context.h"

// #include <algorithm>
// #include <math.h>

// namespace fireworks
// {
//    void invertBox( std::vector<float> &corners )
//    {
//       std::swap( corners[0], corners[9] );
//       std::swap( corners[1], corners[10] );
//       std::swap( corners[2], corners[11] );

//       std::swap( corners[3], corners[6] );
//       std::swap( corners[4], corners[7] );
//       std::swap( corners[5], corners[8] );

//       std::swap( corners[12], corners[21] );
//       std::swap( corners[13], corners[22] );
//       std::swap( corners[14], corners[23] );

//       std::swap( corners[15], corners[18] );
//       std::swap( corners[16], corners[19] );
//       std::swap( corners[17], corners[20] );
//    }

//    void addBox( const std::vector<float> &corners, TEveElement* comp, FWProxyBuilderBase* pb )
//    {
//       TEveBox* eveBox = new TEveBox( "Box" ); 	 
//       eveBox->SetDrawFrame( false );
//       eveBox->SetPickable( true );      
//       eveBox->SetVertices( &corners[0] );

//       pb->setupAddElement( eveBox, comp );
//    }
   
//    void addCircle( double eta, double phi, double radius, const unsigned int nLineSegments, TEveElement* comp, FWProxyBuilderBase* pb )
//    {
//       TEveStraightLineSet* container = new TEveStraightLineSet;
      
//       for( unsigned int iphi = 0; iphi < nLineSegments; ++iphi )
//       {
//          container->AddLine( eta + radius * cos( 2 * M_PI / nLineSegments * iphi ),
//                              phi + radius * sin( 2 * M_PI / nLineSegments * iphi ),
//                              0.01,
//                              eta + radius * cos( 2 * M_PI / nLineSegments * ( iphi + 1 )),
//                              phi + radius * sin( 2 * M_PI / nLineSegments * ( iphi + 1 )),
//                              0.01 );
//       }
//       pb->setupAddElement( container, comp );
//    }

//    void addDashedArrow( double phi, double size, TEveElement* comp, FWProxyBuilderBase* pb )
//    {
//       TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet;
//       marker->SetLineWidth( 1 );
//       marker->SetLineStyle( 2 );
//       marker->AddLine( 0, 0, 0, size * cos( phi ), size * sin( phi ), 0 );
//       marker->AddLine( size * 0.9 * cos( phi + 0.03 ), size * 0.9 * sin( phi + 0.03 ), 0, size * cos( phi ), size * sin( phi ), 0 );
//       marker->AddLine( size * 0.9 * cos( phi - 0.03 ), size * 0.9 * sin( phi - 0.03 ), 0, size * cos( phi ), size * sin( phi ), 0 );
//       pb->setupAddElement( marker, comp );      
//    }
   
//    void addDashedLine( double phi, double theta, double size, TEveElement* comp, FWProxyBuilderBase* pb )
//    {
//       double r( 0 );
//       if( theta < pb->context().caloTransAngle() || M_PI - theta < pb->context().caloTransAngle())
//          r = pb->context().caloZ2() / fabs( cos( theta ));
//       else
//          r = pb->context().caloR1() / sin( theta );
      
//       TEveStraightLineSet* marker = new TEveStraightLineSet;
//       marker->SetLineWidth( 2 );
//       marker->SetLineStyle( 2 );
//       marker->AddLine( r * cos( phi ) * sin( theta ), r * sin( phi ) * sin( theta ), r * cos( theta ),
//                       ( r + size ) * cos( phi ) * sin( theta ), ( r + size ) * sin( phi ) * sin( theta ), ( r + size ) * cos( theta ));
//       pb->setupAddElement( marker, comp );      
//    }
   
//    void addDoubleLines( double phi, TEveElement* comp, FWProxyBuilderBase* pb )
//    {
//       TEveStraightLineSet* mainLine = new TEveStraightLineSet;
//       mainLine->AddLine( -5.191, phi, 0.01, 5.191, phi, 0.01 );
//       pb->setupAddElement( mainLine, comp );
      
//       phi = phi > 0 ? phi - M_PI : phi + M_PI;
//       TEveStraightLineSet* secondLine = new TEveStraightLineSet;
//       secondLine->SetLineStyle( 7 );
//       secondLine->AddLine( -5.191, phi, 0.01, 5.191, phi, 0.01 );
//       pb->setupAddElement( secondLine, comp );      
//    }
   
//    void drawEnergyScaledBox3D( const float* corners, float scale, TEveElement* comp, FWProxyBuilderBase* pb, bool invert )
//    {
//       std::vector<float> scaledCorners( 24 );
//       std::vector<float> centre( 3, 0 );

//       for( unsigned int i = 0; i < 24; i += 3 )
//       {	 
//          centre[0] += corners[i];
//          centre[1] += corners[i + 1];
//          centre[2] += corners[i + 2];
//       }

//       for( unsigned int i = 0; i < 3; ++i )
//          centre[i] *= 1.0f / 8.0f;

//        // Coordinates for a scaled version of the original box
//       for( unsigned int i = 0; i < 24; i += 3 )
//       {	
//          scaledCorners[i] = centre[0] + ( corners[i] - centre[0] ) * scale;
//          scaledCorners[i + 1] = centre[1] + ( corners[i + 1] - centre[1] ) * scale;
//          scaledCorners[i + 2] = centre[2] + ( corners[i + 2] - centre[2] ) * scale;
//       }
      
//       if( invert )
//          invertBox( scaledCorners );

//       addBox( scaledCorners, comp, pb );
//    }
  
//    void drawEtScaledBox3D( const float* corners, float energy, float maxEnergy, TEveElement* comp, FWProxyBuilderBase* pb, bool invert )
//    {
//       std::vector<float> scaledCorners( 24 );
//       std::vector<float> centre( 3, 0 );

//       for( unsigned int i = 0; i < 24; i += 3 )
//       {	 
//          centre[0] += corners[i];
//          centre[1] += corners[i + 1];
//          centre[2] += corners[i + 2];
//       }

//       for( unsigned int i = 0; i < 3; ++i )
//          centre[i] *= 1.0f / 8.0f;

//       TEveVector c( centre[0], centre[1], centre[2] );
//       float scale = energy / maxEnergy * sin( c.Theta());
      
//        // Coordinates for a scaled version of the original box
//       for( unsigned int i = 0; i < 24; i += 3 )
//       {	
//          scaledCorners[i] = centre[0] + ( corners[i] - centre[0] ) * scale;
//          scaledCorners[i + 1] = centre[1] + ( corners[i + 1] - centre[1] ) * scale;
//          scaledCorners[i + 2] = centre[2] + ( corners[i + 2] - centre[2] ) * scale;
//       }
      
//       if( invert )
//          invertBox( scaledCorners );

//       addBox( scaledCorners, comp, pb );
//    }

//    void drawEnergyTower3D( const float* corners, float scale, TEveElement* comp, FWProxyBuilderBase* pb, bool reflect )
//    {
//       std::vector<float> scaledCorners( 24 );
//       for( int i = 0; i < 24; ++i )
//          scaledCorners[i] = corners[i];
//       // Coordinates of a front face scaled 
//       if( reflect )
//       {
//          // We know, that an ES rechit geometry in -Z needs correction. 
//          // The back face is actually its front face.
//          for( unsigned int i = 0; i < 12; i += 3 )
//          {
//             TEveVector diff( corners[i] - corners[i + 12], corners[i + 1] - corners[i + 13], corners[i + 2] - corners[i + 14] );
//             diff.Normalize();
//             diff *= scale;
	    
//             scaledCorners[i] = corners[i] + diff.fX;
//             scaledCorners[i + 1] = corners[i + 1] + diff.fY;
//             scaledCorners[i + 2] = corners[i + 2] + diff.fZ;
//          }
//       } 
//       else
//       {
//          for( unsigned int i = 0; i < 12; i += 3 )
//          {
//             TEveVector diff( corners[i + 12] - corners[i], corners[i + 13] - corners[i + 1], corners[i + 14] - corners[i + 2] );
//             diff.Normalize();
//             diff *= scale;
	    
//             scaledCorners[i] = corners[i + 12];
//             scaledCorners[i + 1] = corners[i + 13];
//             scaledCorners[i + 2] = corners[i + 14];
	    
//             scaledCorners[i + 12] = corners[i + 12] + diff.fX;
//             scaledCorners[i + 13] = corners[i + 13] + diff.fY;
//             scaledCorners[i + 14] = corners[i + 14] + diff.fZ;
//          }
//       }
//       addBox( scaledCorners, comp, pb );
//    }
  
//    void drawEtTower3D( const float* corners, float scale, TEveElement* comp, FWProxyBuilderBase* pb, bool reflect )
//    {
//      std::vector<float> scaledCorners( 24 );
//      for( int i = 0; i < 24; ++i )
//         scaledCorners[i] = corners[i];
//       // Coordinates of a front face scaled 
//       if( reflect )
//       {
//          // We know, that an ES rechit geometry in -Z needs correction. 
//          // The back face is actually its front face.
//          for( unsigned int i = 0; i < 12; i += 3 )
//          {
//             TEveVector diff( corners[i] - corners[i + 12], corners[i + 1] - corners[i + 13], corners[i + 2] - corners[i + 14] );
//             diff.Normalize();
//             diff *= ( scale * sin( diff.Theta()));
	    
//             scaledCorners[i] = corners[i] + diff.fX;
//             scaledCorners[i + 1] = corners[i + 1] + diff.fY;
//             scaledCorners[i + 2] = corners[i + 2] + diff.fZ;
//          }
//       } 
//       else
//       {
//          for( unsigned int i = 0; i < 12; i += 3 )
//          {
//             TEveVector diff( corners[i + 12] - corners[i], corners[i + 13] - corners[i + 1], corners[i + 14] - corners[i + 2] );
//             diff.Normalize();
//             diff *= ( scale * sin( diff.Theta()));
	    
//             scaledCorners[i] = corners[i + 12];
//             scaledCorners[i + 1] = corners[i + 13];
//             scaledCorners[i + 2] = corners[i + 14];
	    
//             scaledCorners[i + 12] = corners[i + 12] + diff.fX;
//             scaledCorners[i + 13] = corners[i + 13] + diff.fY;
//             scaledCorners[i + 14] = corners[i + 14] + diff.fZ;
//          }
//       }
//       addBox( scaledCorners, comp, pb );
//    }
// }
