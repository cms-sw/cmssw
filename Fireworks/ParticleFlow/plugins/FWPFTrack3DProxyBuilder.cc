#include "FWPFTrack3DProxyBuilder.h"

//______________________________________________________________________________________________________________________________________________
void
FWPFTrack3DProxyBuilder::build( const reco::Track &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   if( context().getField()->getSource() == FWMagField::kNone )
   {
      if( fabs( iData.eta() ) < 2.0 && iData.pt() > 0.5 && iData.pt() < 30 )
      {
         double estimate = fw::estimate_field( iData, true );
         if( estimate >= 0 ) context().getField()->guessField( estimate );
      }
   }

   TEveTrack *trk = m_trackUtils->getTrack( iData );

   const Float_t *points = trk->GetP();
   TEvePointSet *ps = new TEvePointSet();
   bool draw = false;

   for( Int_t i = 0; i < trk->GetN(); ++i )
   {
      int j = i * 3;
      TEveVector v1 = TEveVector( points[j], points[j+1], points[j+2] );

      if( m_trackUtils->checkIntersect( v1, context().caloR1( false ) ) )
      {
         TEveVector v2 = TEveVector( points[j-3], points[j-2], points[j-1] );
         TEveVector xyPoint = m_trackUtils->lineCircleIntersect( v1, v2, context().caloR1( false ) );
         TEveVector zPoint;
         if( v1.fZ < 0 )
            zPoint = TEveVector( xyPoint.fX, xyPoint.fY, v1.fZ - 50.f );
         else
            zPoint = TEveVector( xyPoint.fX, xyPoint.fY, v1.fZ + 50.f );

         TEveVector vec = m_trackUtils->lineLineIntersect( v1, v2, xyPoint, zPoint );
         ps->SetNextPoint( vec.fX, vec.fY, vec.fZ );
         draw = true;
         break;   // Only care about the first point that meets this condition
      }
      else if( fabs( v1.fZ ) >= context().caloZ1( false ) )
      {
         TEveVector p1, p2;
         TEveVector vec, v2 = TEveVector( points[j-3], points[j-2], points[j-1] );
         float z, y = m_trackUtils->linearInterpolation( v2, v1, context().caloZ1( false ) );

         if( v2.fZ > 0 )
            z = context().caloZ1( false );
         else
            z = context().caloZ1( false ) * -1;

         p1 = TEveVector( v2.fX + 50, y, z );
         p2 = TEveVector( v2.fX - 50, y, z );
         vec = m_trackUtils->lineLineIntersect( v1, v2, p1, p2 );

         ps->SetNextPoint( vec.fX, vec.fY, vec.fZ );
         draw = true;
         break;   // Only care about the first point that meets this condition
      }
   }

   if( draw )
   {
      int i = ( trk->GetN() - 1 ) * 3;
      TEveVector vec = TEveVector( points[i], points[i+1], points[i+2] );

      if( m_trackUtils->checkIntersect( vec, 177.f - 1 ) )
         ps->SetNextPoint( vec.fX, vec.fY, vec.fZ );
      else if( fabs( vec.fZ ) >= context().caloZ1( false ) )
         ps->SetNextPoint( vec.fX, vec.fY, vec.fZ );
   }

   setupAddElement( trk, &oItemHolder );
   if( draw )
      setupAddElement( ps, &oItemHolder );
   else
      delete ps;
}

//______________________________________________________________________________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFTrack3DProxyBuilder, reco::Track, "PF Tracks", FWViewType::k3DBit );
