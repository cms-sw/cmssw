#include "FWPFTrackRPZProxyBuilder.h"

//______________________________________________________________________________
void
FWPFTrackRPZProxyBuilder::build( const reco::Track &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   TEveTrack *trk = m_trackUtils->setupRPZTrack( iData );
   TEvePointSet *ps = m_trackUtils->getCollisionMarkers( trk );
   setupAddElement( trk, &oItemHolder );
   if( ps->GetN() != 0 )
      setupAddElement( ps, &oItemHolder );
   else
      delete ps;   
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFTrackRPZProxyBuilder, reco::Track, "PF Tracks", FWViewType::kRhoPhiPFBit  | FWViewType::kRhoZBit );
