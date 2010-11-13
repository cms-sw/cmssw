#include "LegoRecHit.h"

void
LegoRecHit::setupEveBox( TEveBox *eveBox, size_t numCorners, const std::vector<TEveVector> &corners )
{
   for( size_t i = 0; i < numCorners; i++ )
      eveBox->SetVertex( i, corners[i] );
   eveBox->SetPickable( true );
   eveBox->SetDrawFrame( true );
   eveBox->SetLineWidth( 1.0 );
   eveBox->SetLineColor( kBlack );
}

std::vector<TEveVector>
LegoRecHit::convertToTower( const std::vector<TEveVector> &corners, float e_t, float scale )
{
   std::vector<TEveVector> towerCorners = corners;

   for( size_t i = 0; i < 4; i++ )
   {
      TEveVector diff = towerCorners[i+4] - towerCorners[i];
      diff.Normalize();
      towerCorners[i+4] = towerCorners[i];
      towerCorners[i] = towerCorners[i] + ( diff * ( e_t / scale ) );
   }

   return towerCorners;
}

LegoRecHit::LegoRecHit( size_t numCorners, const std::vector<TEveVector> &corners, TEveElement *comp, FWProxyBuilderBase *pb, float e_t, float scale )
{
   tower = new TEveBox("EcalRecHitTower");
   std::vector<TEveVector> towerCorners = corners;

   towerCorners = convertToTower( towerCorners, e_t, scale );

   setupEveBox( tower, numCorners, towerCorners );
   pb->setupAddElement( tower, comp );


//--------------------------------------------------------------
// Temporary solution
//--------------------------------------------------------------
   std::vector<TEveVector> squareCorners(4);
   float centre[2] = {0, 0};
   int i = 0;

   lineSet = new TEveLine("Square");
   lineSet->SetLineWidth( 2.0 );

   for( i = 0; i < 4; i++ )
   {
      centre[0] += towerCorners[i].fX;
      centre[1] += towerCorners[i].fY;      // Gather data for centre point
      squareCorners[i] = towerCorners[i];
      squareCorners[i].fZ += 0.001;         // Square has to be slightly above tower for visibility
   }


   centre[0] *= 1.0f / 4.0f;               // Calculate centre point
   centre[1] *= 1.0f / 4.0f;

   for( i = 0; i < 4; i++ )
   {   // Do the actual scaling;
      squareCorners[i].fX = centre[0] + ( ( squareCorners[i].fX - centre[0] ) * ( log( 1 + e_t ) / log(scale) ) );
      squareCorners[i].fY = centre[1] + ( ( squareCorners[i].fY - centre[1] ) * ( log( 1 + e_t ) / log(scale) ) );
   }

   for( int i = 0; i < 4; i++ )
      lineSet->SetNextPoint( squareCorners[i].fX, squareCorners[i].fY, squareCorners[i].fZ );
   lineSet->SetNextPoint( squareCorners[0].fX, squareCorners[0].fY, squareCorners[0].fZ );

   if( e_t == scale )                     // Highest e_t in the current data collection
   {
      for( int i = 0; i < 2; i++ )
      {
         lineSet->SetNextPoint( squareCorners[i].fX, squareCorners[i].fY, squareCorners[i].fZ );
         lineSet->SetNextPoint( squareCorners[i+2].fX, squareCorners[i+2].fY, squareCorners[i+2].fZ );
      }
   }

   pb->setupAddElement( lineSet, comp );
}
