#include "FWPFTrackProxyBuilder.h"

//______________________________________________________________________________________________________________________________________________
TEveTrack *
FWPFTrackProxyBuilder::getTrack( unsigned int id, const reco::Track &iData )
{
   if( id < tracks.size() )   // This id is already known
      return tracks[id];

   // Only gets here if id is already known
   TEveTrackPropagator *propagator = ( !iData.extra().isAvailable() ) ? context().getTrackerTrackPropagator() : context().getTrackPropagator();
   propagator->SetMaxR( 177.f );

   TEveRecTrack t;
   t.fBeta = 1.;
   t.fP = TEveVector( iData.px(), iData.py(), iData.pz() );
   t.fV = TEveVector( iData.vertex().x(), iData.vertex().y(), iData.vertex().z() );
   t.fSign = iData.charge();
   TEveTrack* trk = new TEveTrack( &t, propagator );
   trk->MakeTrack();
   tracks.push_back( trk );

   return trk;
}

//______________________________________________________________________________________________________________________________________________
void
FWPFTrackProxyBuilder::cleanLocal()
{
   // STILL NEED TO ADD A DELETE WITHOUT CAUSING SEG FAULT!!!
   tracks.clear();
}

//______________________________________________________________________________________________________________________________________________
void
FWPFTrackProxyBuilder::buildViewType( const reco::Track &iData, unsigned int iIndex, TEveElement &oItemHolder, 
                                      FWViewType::EType viewType , const FWViewContext* )
{
   const FWEventItem::ModelInfo &info = item()->modelInfo( iIndex );

   if( info.displayProperties().isVisible() )
   {
      if( context().getField()->getSource() == FWMagField::kNone )
      {
        if( fabs( iData.eta() ) < 2.0 && iData.pt() > 0.5 && iData.pt() < 30 )
        {
         double estimate = fw::estimate_field( iData, true );
         if( estimate >= 0 ) context().getField()->guessField( estimate );
        }
      }

      if( viewType == FWViewType::kRhoPhiPF )
      {
         TEveTrack *trk = getTrack( iIndex, iData );
         setupAddElement( trk, &oItemHolder );     // Seg fault here!!!
      }
      else if( viewType == FWViewType::kRhoPhi )
      {
         //TEveTrack *trk = getTrack( iIndex, iData );
         //setupAddElement( trk, &oItemHolder );
      }
      else if( viewType == FWViewType::kLegoPFECAL )
      {
         TEveTrack *trk = getTrack( iIndex, iData );
         std::vector<TEveVector> trackPoints(0);
         //std::vector<TEveVector> trackPoints( trk->GetN() );
         const Float_t *points = trk->GetP();

         for( Int_t i = 1; i < trk->GetN(); ++i )
         {
            int j = i * 3;
            TEveVector temp = TEveVector( points[j], points[j+1], points[j+2] );
            TEveVector vec = TEveVector( temp.Eta(), temp.Phi(), 0.001 );

            //trackPoints[i] = vec;
            trackPoints.push_back( vec );
         }

         for( unsigned int i = 0; i < trackPoints.size() - 1; ++i )
         {
            if( ( trackPoints[i+1].fY - trackPoints[i].fY ) > 1 )
               trackPoints[i+1].fY -= TMath::TwoPi();
   
            if( ( trackPoints[i].fY - trackPoints[i+1].fY ) > 1 )
               trackPoints[i+1].fY += TMath::TwoPi();
         }

         TEveLine *newTrack = new TEveLine();
         for( unsigned int i = 0; i < trackPoints.size(); ++i )
            newTrack->SetNextPoint( trackPoints[i].fX, trackPoints[i].fY, 0.001 );

         setupAddElement( newTrack, &oItemHolder );
      }
   }
}

//______________________________________________________________________________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFTrackProxyBuilder, reco::Track, "PF Tracks", FWViewType::kRhoPhiPFBit | FWViewType::kLegoPFECALBit );
