#include "Fireworks/ParticleFlow/interface/FWPFTrackUtils.h"

//______________________________________________________________________________________________________________________________________________
FWPFTrackUtils::FWPFTrackUtils() : m_trackerTrackPropagator(0), m_trackPropagator(0), m_magField(0)
{
   float caloTransAngle = 1.479;

	// ECAL
   m_caloR1 = 129;		// Centres of front faces of the crystals in supermodules (1.29m) - Barrel
   m_caloZ1 = 303.353;	// Longitudinal distance between the interaction point and last tracker layer - Endcap

	// HCAL
	m_caloR2 = 177.7;		// Longitudinal profile in the barrel (inner radius) - Barrel
	m_caloR3 = 287.65;	// Longitudinal profile in the barrel (outer radius) - Barrel
	m_caloZ2 = 400.458;	// Longitudinal distance between the interaction point and the endcap envelope - Endcap

   m_magField = new FWMagField();

   // Common propagator, helix stepper
   m_trackPropagator = new TEveTrackPropagator();
   m_trackPropagator->SetMagFieldObj( m_magField, false );
   m_trackPropagator->SetMaxR( m_caloR3 );
   m_trackPropagator->SetMaxZ( m_caloZ2 );
   m_trackPropagator->SetDelta( 0.01 );
   m_trackPropagator->SetProjTrackBreaking( TEveTrackPropagator::kPTB_UseLastPointPos );
   m_trackPropagator->SetRnrPTBMarkers( kTRUE );
   m_trackPropagator->IncDenyDestroy();

   // Tracker propagator
   m_trackerTrackPropagator = new TEveTrackPropagator();
   m_trackerTrackPropagator->SetStepper( TEveTrackPropagator::kRungeKutta );
   m_trackerTrackPropagator->SetMagFieldObj( m_magField, false );
   m_trackerTrackPropagator->SetDelta( 0.01 );
   m_trackerTrackPropagator->SetMaxR( m_caloR3 );
   m_trackerTrackPropagator->SetMaxZ( m_caloZ2 );
   m_trackerTrackPropagator->SetProjTrackBreaking( TEveTrackPropagator::kPTB_UseLastPointPos );
   m_trackerTrackPropagator->SetRnrPTBMarkers( kTRUE );
   m_trackerTrackPropagator->IncDenyDestroy();
}

//______________________________________________________________________________________________________________________________________________
TEveTrack *
FWPFTrackUtils::getTrack( const reco::Track &iData )
{
   TEveTrackPropagator *propagator = ( !iData.extra().isAvailable() ) ? m_trackerTrackPropagator : m_trackPropagator;

   TEveRecTrack t;
   t.fBeta = 1;
   t.fP = TEveVector( iData.px(), iData.py(), iData.pz() );
   t.fV = TEveVector( iData.vertex().x(), iData.vertex().y(), iData.vertex().z() );
   t.fSign = iData.charge();
   TEveTrack *trk = new TEveTrack( &t, propagator );
   trk->MakeTrack();

   return trk;
}

//______________________________________________________________________________________________________________________________________________
TEveVector
FWPFTrackUtils::lineCircleIntersect( const TEveVector &v1, const TEveVector &v2, float r )
{
   // Definitions
   float x, y;
   float dx = v1.fX - v2.fX;
   float dy = v1.fY - v2.fY;
   float dr = sqrt( ( dx * dx ) + ( dy * dy ) );
   float D = ( ( v2.fX * v1.fY ) - ( v1.fX * v2.fY ) );

   float rtDescrim = sqrt( ( ( r * r ) * ( dr * dr ) ) - ( D * D ) );

   if( dy < 0 )   // Going down
   {
      x = ( D * dy ) - ( ( sgn(dy) * dx ) * rtDescrim );
      x /= ( dr * dr );

      y = ( -D * dx ) - ( fabs( dy ) * rtDescrim );
      y /= ( dr * dr );
   }
   else
   {

      x = ( D * dy ) + ( ( sgn(dy) * dx ) * rtDescrim );
      x /= ( dr * dr );

      y = ( -D * dx ) + ( fabs( dy ) * rtDescrim );
      y /= ( dr * dr );
   }

   TEveVector result = TEveVector( x, y, 0.001 );
   return result;
}

//______________________________________________________________________________________________________________________________________________
TEveVector
FWPFTrackUtils::lineLineIntersect( const TEveVector &p1, const TEveVector &p2, const TEveVector &p3, const TEveVector &p4 )
{
   TEveVector a = p2 - p1;
   TEveVector b = p4 - p3;
   TEveVector c = p3 - p1;
   TEveVector result;
   float s, val;

   s = dot( cross( c, b ), cross( a, b ) );  
   val = dot( cross( a, b ), cross( a, b ) );
   s /= val;
   
   result = p1 + ( a * s );
   return result; 
}

//______________________________________________________________________________________________________________________________________________
TEveVector
FWPFTrackUtils::cross( const TEveVector &v1, const TEveVector &v2 )
{
   TEveVector vec;

   vec.fX = ( v1.fY * v2.fZ ) - ( v1.fZ * v2.fY );
   vec.fY = ( v1.fZ * v2.fX ) - ( v1.fX * v2.fZ );
   vec.fZ = ( v1.fX * v2.fY ) - ( v1.fY * v2.fX );

   return vec;
}

//______________________________________________________________________________________________________________________________________________
float
FWPFTrackUtils::linearInterpolation( const TEveVector &p1, const TEveVector &p2, float z )
{
   float y;

   y = ( ( z - fabs( p1.fZ ) ) * p2.fY ) + ( ( fabs( p2.fZ ) - z ) * p1.fY );
   y /= ( fabs( p2.fZ) - fabs( p1.fZ ) );

   return y;
}

//______________________________________________________________________________________________________________________________________________
float
FWPFTrackUtils::dot( const TEveVector &v1, const TEveVector &v2 )
{
   float result = ( v1.fX * v2.fX ) + ( v1.fY * v2.fY ) + ( v1.fZ * v1.fZ );

   return result;
}

//______________________________________________________________________________________________________________________________________________
float
FWPFTrackUtils::sgn( float val )
{
   return ( val < 0 ) ? -1 : 1;
}

//______________________________________________________________________________________________________________________________________________
bool
FWPFTrackUtils::checkIntersect( const TEveVector &p, float r )
{
   float h = sqrt( ( p.fX * p.fX ) + ( p.fY * p.fY ) );
   
   if( h >= r )
      return true;

   return false;
}
