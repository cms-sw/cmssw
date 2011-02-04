#ifndef _FWPFTRACKLEGOPROXYBUILDER_H_
#define _FWPFTRACKLEGOPROXYBUILDER_H_

//
// Package:					Particle Flow
// Class:					FWPFTrackLegoProxyBuilder
// Original Author:		Simon Harris
//

// System include files
#include "TEveStraightLineSet.h"

// User include files
#include "FWPFTrackBaseProxyBuilder.h"

//-----------------------------------------------------------------------------
// FWPFTrackLegoProxyBuilder
//-----------------------------------------------------------------------------

class FWPFTrackLegoProxyBuilder : public FWPFTrackBaseProxyBuilder
{
   public:
	// ---------------- Constructor(s)/Destructor ---------------------
      FWPFTrackLegoProxyBuilder(){}
      virtual ~FWPFTrackLegoProxyBuilder(){}

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFTrackLegoProxyBuilder( const FWPFTrackLegoProxyBuilder& );
      const FWPFTrackLegoProxyBuilder& operator=( const FWPFTrackLegoProxyBuilder& );

	// ----------------------- Functionality ---------------------------
      virtual void build( const reco::Track &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc );
};
#endif

//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
