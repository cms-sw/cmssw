#include "FWPFTrackLegoProxyBuilder.h"

//______________________________________________________________________________________________________________________________________________
void
FWPFTrackLegoProxyBuilder::build( const reco::Track &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   // Declarations
   int wraps[2] = { -1, -1 };
   TEveTrack *trk = m_trackUtils->getTrack( iData );
   std::vector<TEveVector> trackPoints( trk->GetN() - 1 );
   const Float_t *points = trk->GetP();
   TEveStraightLineSet *legoTrack = new TEveStraightLineSet();

   if( context().getField()->getSource() == FWMagField::kNone )
   {
      if( fabs( iData.eta() ) < 2.0 && iData.pt() > 0.5 && iData.pt() < 30 )
      {
         double estimate = fw::estimate_field( iData, true );
         if( estimate >= 0 ) context().getField()->guessField( estimate );
      }
   }

   // Convert to Eta/Phi and store in vector
   for( Int_t i = 1; i < trk->GetN(); ++i )
   {
      int j = i * 3;
      TEveVector temp = TEveVector( points[j], points[j+1], points[j+2] );
      TEveVector vec = TEveVector( temp.Eta(), temp.Phi(), 0.001 );

      trackPoints[i-1] = vec;
   }

   // Add first point to ps if necessary
   for( Int_t i = 1; i < trk->GetN(); ++i )
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
         legoTrack->AddMarker( vec.Eta(), vec.Phi(), 0.001, 0 );

         wraps[0] = i;        // There is now a chance that the track will also reach the HCAL radius
         break;               // Only want the first point that matches the check condition
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

         legoTrack->AddMarker( vec.Eta(), vec.Phi(), 0.001, 0 );
         wraps[0] = i;
         break;   // Only care about the first point that meets this condition
      }
   }

   if( wraps[0] != -1 )
   {
      int i = ( trk->GetN() - 1 ) * 3;
      int j = trk->GetN() - 2;   // This is equal to the last index in trackPoints vector
      TEveVector vec = TEveVector( points[i], points[i+1], points[i+2] );

      if( m_trackUtils->checkIntersect( vec, 177.f - 1 ) )
      {
         legoTrack->AddMarker( vec.Eta(), vec.Phi(), 0.001, 1 );

         wraps[1] = j;
      }
      else if( fabs( vec.fZ ) >= context().caloZ1( false ) )
      {
         legoTrack->AddMarker( vec.Eta(), vec.Phi(), 0.001, 1 );
         wraps[1] = j;
      }
   }

   // Handle phi wrapping
   for( int i = 0; i < static_cast<int>( trackPoints.size() - 1 ); ++i )
   {
      if( ( trackPoints[i+1].fY - trackPoints[i].fY ) > 1 )
      {
         trackPoints[i+1].fY -= TMath::TwoPi();

         if( i == wraps[0] )
         {
            TEveChunkManager::iterator mi( legoTrack->GetMarkerPlex() );
            mi.next();  // First point
            TEveStraightLineSet::Marker_t &m = * ( TEveStraightLineSet::Marker_t* ) mi();
            m.fV[0] = trackPoints[i+1].fX; m.fV[1] = trackPoints[i+1].fY; m.fV[2] = 0.001;
         }
      }

      if( ( trackPoints[i].fY - trackPoints[i+1].fY ) > 1 )
      {
         trackPoints[i+1].fY += TMath::TwoPi();
         if( i == wraps[0] )
         {
            TEveChunkManager::iterator mi( legoTrack->GetMarkerPlex() );
            mi.next();  // First point
            TEveStraightLineSet::Marker_t &m = * ( TEveStraightLineSet::Marker_t* ) mi();
            m.fV[0] = trackPoints[i+1].fX; m.fV[1] = trackPoints[i+1].fY; m.fV[2] = 0.001;
         }
      }
   }

   int end = static_cast<int>( trackPoints.size() - 1 );
   if( wraps[1] == end )
   {
      TEveChunkManager::iterator mi( legoTrack->GetMarkerPlex() );
      mi.next(); mi.next();   // Second point
      TEveStraightLineSet::Marker_t &m = * ( TEveStraightLineSet::Marker_t* ) mi();
      m.fV[0] = trackPoints[end].fX; m.fV[1] = trackPoints[end].fY; m.fV[2] = 0.001;
   }

   // Set points on TEveLineSet object ready for displaying
   for( unsigned int i = 0;i < trackPoints.size() - 1; ++i )
      legoTrack->AddLine( trackPoints[i], trackPoints[i+1] );

   legoTrack->SetDepthTest( false );
   legoTrack->SetMarkerStyle( 4);
   legoTrack->SetMarkerSize( 1 );

   setupAddElement( legoTrack, &oItemHolder );
   delete trk;    // Release memory that is no longer required
}

//______________________________________________________________________________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFTrackLegoProxyBuilder, reco::Track, "PF Tracks", FWViewType::kLegoPFECALBit | FWViewType::kLegoBit );
