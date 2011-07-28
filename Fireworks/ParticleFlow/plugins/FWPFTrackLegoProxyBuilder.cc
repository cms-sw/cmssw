#include "FWPFTrackLegoProxyBuilder.h"

//______________________________________________________________________________
void
FWPFTrackLegoProxyBuilder::build( const reco::Track &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   TEveStraightLineSet *legoTrack = m_trackUtils->setupLegoTrack( iData );
   legoTrack->SetRnrMarkers( true );
   setupAddElement( legoTrack, &oItemHolder ); 
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFTrackLegoProxyBuilder, reco::Track, "PF Tracks", FWViewType::kLegoPFECALBit | FWViewType::kLegoBit );
