#include "FWPFTrackLegoProxyBuilder.h"

//______________________________________________________________________________
void
FWPFTrackLegoProxyBuilder::build( const reco::Track &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   FWPFTrackUtils *utils = new FWPFTrackUtils();
   TEveStraightLineSet *legoTrack = utils->setupLegoTrack( iData );
   setupAddElement( legoTrack, &oItemHolder );
   delete utils;
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFTrackLegoProxyBuilder, reco::Track, "PF Tracks", FWViewType::kLegoPFECALBit | FWViewType::kLegoBit );
