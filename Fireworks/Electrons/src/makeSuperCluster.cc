// -*- C++ -*-
//
// Package:     Electrons
// Class  :     makeSuperCluster
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Dec  5 15:32:33 EST 2008
// $Id: makeSuperCluster.cc,v 1.13 2010/09/16 15:42:21 yana Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"

// user include files
#include "Fireworks/Electrons/interface/makeSuperCluster.h"

#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h" 

namespace fireworks {
bool makeRhoPhiSuperCluster( FWProxyBuilderBase* pb,
                             const reco::SuperClusterRef& iCluster,
                             float iPhi,
                             TEveElement& oItemHolder )
{
   if( !iCluster.isAvailable()) return false;
   TEveGeoManagerHolder gmgr( TEveGeoShape::GetGeoMangeur());

   std::vector< std::pair<DetId, float> > detids = iCluster->hitsAndFractions();
   std::vector<double> phis;
   for( std::vector<std::pair<DetId, float> >::const_iterator id = detids.begin(), end = detids.end(); id != end; ++id )
   {
      const float* corners = pb->context().getGeom()->getCorners( id->first.rawId());
      if( corners != 0 )
      {
         std::vector<float> centre( 3, 0 );

         for( unsigned int i = 0; i < 24; i += 3 )
         {	 
            centre[0] += corners[i];
            centre[1] += corners[i + 1];
            centre[2] += corners[i + 2];
         }
       
         phis.push_back( TEveVector( centre[0], centre[1], centre[2] ).Phi());
      }
   }
   std::pair<double,double> phiRange = fireworks::getPhiRange( phis, iPhi );
   const double r = pb->context().caloR1();
   TGeoBBox *sc_box = new TGeoTubeSeg( r - 2, r , 1,
                                       phiRange.first * 180 / M_PI - 0.5,
                                       phiRange.second * 180 / M_PI + 0.5 ); // 0.5 is roughly half size of a crystal
   TEveGeoShape *sc = fireworks::getShape( "supercluster", sc_box, pb->item()->defaultDisplayProperties().color());
   sc->SetPickable( kTRUE );
   pb->setupAddElement( sc, &oItemHolder );
   return true;
}

bool makeRhoZSuperCluster( FWProxyBuilderBase* pb,
                           const reco::SuperClusterRef& iCluster,
                           float iPhi,
                           TEveElement& oItemHolder )
{
   if( !iCluster.isAvailable()) return false;
   TEveGeoManagerHolder gmgr( TEveGeoShape::GetGeoMangeur());
   double theta_max = 0;
   double theta_min = 10;
   std::vector<std::pair<DetId, float> > detids = iCluster->hitsAndFractions();
   for( std::vector<std::pair<DetId, float> >::const_iterator id = detids.begin(), end = detids.end(); id != end; ++id )
   {
      const float* corners = pb->context().getGeom()->getCorners( id->first.rawId());
      if( corners != 0 )
      {
         std::vector<float> centre( 3, 0 );

         for( unsigned int i = 0; i < 24; i += 3 )
         {	 
            centre[0] += corners[i];
            centre[1] += corners[i + 1];
            centre[2] += corners[i + 2];
         }

         double theta = TEveVector( centre[0], centre[1], centre[2] ).Theta();
         if( theta > theta_max ) theta_max = theta;
         if( theta < theta_min ) theta_min = theta;
      }
   }
   // expand theta range by the size of a crystal to avoid segments of zero length
   bool barrel = true; 
   if ((theta_max > 0 && theta_max <  pb->context().caloTransAngle()) || 
       ( theta_min > (TMath::Pi() -pb->context().caloTransAngle())) )
   {
        barrel = false; 
   }
 
   double z_ecal = barrel ? pb->context().caloZ1() : pb->context().caloZ2();
   double r_ecal = barrel ? pb->context().caloR1() : pb->context().caloR2();

   fireworks::addRhoZEnergyProjection( pb, &oItemHolder, r_ecal-1, z_ecal-1,
				       theta_min - 0.003, theta_max + 0.003,
				       iPhi );

   return true;
}

}
