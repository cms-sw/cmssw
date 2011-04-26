#include "Fireworks/ParticleFlow/interface/FWPFClusterRPZUtils.h"

//______________________________________________________________________________
float
FWPFClusterRPZUtils::calculateEt( const reco::PFCluster &cluster, float e )
{
   float et = 0.f;
   TEveVector vec;

   vec.fX = cluster.x();
   vec.fY = cluster.y();
   vec.fZ = cluster.z();

   vec.Normalize();
   vec *= e;
   et = vec.Perp();

   return et;
}

//______________________________________________________________________________
TEveScalableStraightLineSet *
FWPFClusterRPZUtils::buildRhoPhiClusterLineSet( const reco::PFCluster &cluster, const FWViewContext *vc, float r )
{
   float energy, et;

   energy = cluster.energy();
   et = calculateEt( cluster, energy );

   return buildRhoPhiClusterLineSet( cluster, vc, energy, et, r );
}

//______________________________________________________________________________
TEveScalableStraightLineSet *
FWPFClusterRPZUtils::buildRhoPhiClusterLineSet( const reco::PFCluster &cluster, const FWViewContext *vc, 
                                                float e, float et, float r )
{
   TEveScalableStraightLineSet *ls = new TEveScalableStraightLineSet( "rhophiCluster" );
   TEveVector vec;
   float size = 1.f; // Stored in scale
   double phi;

   vec = TEveVector( cluster.x(), cluster.y(), cluster.z() );
   phi = vec.Phi();

   FWViewEnergyScale *energyScale = vc->getEnergyScale();
   ls->SetLineWidth( 4 );

   ls->SetScaleCenter( r * cos( phi ), r * sin( phi ), 0 );
   ls->AddLine( r * cos( phi ), r * sin( phi ), 0, ( r + size ) * cos( phi ), ( r + size ) * sin( phi ), 0 );
   ls->SetScale( energyScale->getScaleFactor3D() * ( energyScale->getPlotEt() ? et : e ) );

   return ls;
}

//______________________________________________________________________________
TEveScalableStraightLineSet *
FWPFClusterRPZUtils::buildRhoZClusterLineSet( const reco::PFCluster &cluster, const FWViewContext *vc,
                                              float caloTransAngle, float r, float z )
{
   float energy, et;

   energy = cluster.energy();
   et = calculateEt( cluster, energy );

   return buildRhoZClusterLineSet( cluster, vc, caloTransAngle, energy, et, r, z );
}

//______________________________________________________________________________
TEveScalableStraightLineSet *
FWPFClusterRPZUtils::buildRhoZClusterLineSet( const reco::PFCluster &cluster, const FWViewContext *vc,
                                              float caloTransAngle, float e, float et, float r, float z )
{
      float size = 1.f;       // Stored in scale
      float offr = 4;
      float ecalZ = z + offr / tan( caloTransAngle );
      double theta, phi;
      double rad(0);
      TEveVector vec;
      TEveScalableStraightLineSet *ls = new TEveScalableStraightLineSet( "rhoZCluster" );

      vec = TEveVector( cluster.x(), cluster.y(), cluster.z() ); 
      phi = vec.Phi();
      theta =  vec.Theta();

      FWViewEnergyScale *caloScale = vc->getEnergyScale();
      ls->SetLineWidth( 4 );

      if ( theta < caloTransAngle || TMath::Pi() - theta < caloTransAngle )
         rad = ecalZ / fabs( cos( theta ) );
      else
         rad = r / sin( theta );

      ls->SetScaleCenter( 0., ( phi > 0 ? rad * fabs( sin( theta ) ) : -rad * fabs( sin( theta ) ) ), rad * cos( theta ) );
      ls->AddLine( 0., ( phi > 0 ? rad * fabs( sin( theta ) ) : -rad * fabs( sin( theta ) ) ), rad * cos( theta ),
                   0., ( phi > 0 ? ( rad + size ) * fabs( sin ( theta ) ) : -( rad + size ) * fabs( sin( theta) ) ), 
                   ( rad + size ) * cos( theta ) );
      ls->SetScale( caloScale->getScaleFactor3D() * ( caloScale->getPlotEt() ? et : e ) );

      return ls;
}

