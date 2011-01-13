#include "FWPFTrackProxyBuilder.h"

FWPFTrackProxyBuilder::FWPFTrackProxyBuilder() : m_trackerTrackPropagator(0), m_trackPropagator(0), m_magField(0)
{
   float caloTransEta = 1.479;
   float caloTransAngle = 2*atan(exp(-caloTransEta));
   float caloR = 290*tan(caloTransAngle);
   float propagatorOffR = 5;
   float propagatorOffZ = propagatorOffR * ( 290 / caloR );
   m_magField = new FWMagField();

   // common propagator, helix stepper
   m_trackPropagator = new TEveTrackPropagator();
   m_trackPropagator->SetMagFieldObj( m_magField, false );
   m_trackPropagator->SetDelta( 0.01 );
   m_trackPropagator->SetMaxR( 177.f );
   m_trackPropagator->SetMaxZ( 290 - propagatorOffZ );
   m_trackPropagator->SetProjTrackBreaking( TEveTrackPropagator::kPTB_UseLastPointPos );
   m_trackPropagator->SetRnrPTBMarkers( kTRUE );
   m_trackPropagator->IncDenyDestroy();

   // tracker propagator
   m_trackerTrackPropagator = new TEveTrackPropagator();
   m_trackerTrackPropagator->SetStepper( TEveTrackPropagator::kRungeKutta );
   m_trackerTrackPropagator->SetMagFieldObj( m_magField, false );
   m_trackerTrackPropagator->SetDelta( 0.01 );
   m_trackerTrackPropagator->SetMaxR( 177.f );
   m_trackerTrackPropagator->SetMaxZ( 290 - propagatorOffZ );
   m_trackerTrackPropagator->SetProjTrackBreaking( TEveTrackPropagator::kPTB_UseLastPointPos );
   m_trackerTrackPropagator->SetRnrPTBMarkers( kTRUE );
   m_trackerTrackPropagator->IncDenyDestroy();
}

//______________________________________________________________________________________________________________________________________________
TEveTrack *
FWPFTrackProxyBuilder::getTrack( unsigned int id, const reco::Track &iData )
{
   if( id < tracks.size() )   // This id is already known
      return tracks[id];

   // Only gets here if id is already known
   TEveTrackPropagator *propagator = ( !iData.extra().isAvailable() ) ? m_trackerTrackPropagator : m_trackPropagator;

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

      TEveTrack *trk = getTrack( iIndex, iData );

      if( viewType == FWViewType::kRhoPhiPF )
         setupAddElement( trk, &oItemHolder );
      else
      {
         TEveTrack *trk = getTrack( iIndex, iData );
         std::vector<TEveVector> trackPoints( trk->GetN() - 1 );
         const Float_t *points = trk->GetP();

         for( Int_t i = 1; i < trk->GetN(); ++i )
         {
            int j = i * 3;
            TEveVector temp = TEveVector( points[j], points[j+1], points[j+2] );
            TEveVector vec = TEveVector( temp.Eta(), temp.Phi(), 0.001 );

            trackPoints[i-1] = vec;
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
REGISTER_FWPROXYBUILDER( FWPFTrackProxyBuilder, reco::Track, "PF Tracks", FWViewType::kRhoPhiPFBit | FWViewType::kLegoPFECALBit | FWViewType::kLegoBit );
