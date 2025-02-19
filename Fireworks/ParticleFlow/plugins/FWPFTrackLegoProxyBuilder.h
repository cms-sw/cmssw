#ifndef _FWPFTRACKLEGOPROXYBUILDER_H_
#define _FWPFTRACKLEGOPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFTrackLegoProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include "TEveStraightLineSet.h"

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/ParticleFlow/interface/FWPFTrackUtils.h"

//-----------------------------------------------------------------------------
// FWPFTrackLegoProxyBuilder
//-----------------------------------------------------------------------------
class FWPFTrackLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track>
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFTrackLegoProxyBuilder(){}
      virtual ~FWPFTrackLegoProxyBuilder(){}

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFTrackLegoProxyBuilder( const FWPFTrackLegoProxyBuilder& );
      const FWPFTrackLegoProxyBuilder& operator=( const FWPFTrackLegoProxyBuilder& );

   // --------------------- Member Functions --------------------------
      virtual void build( const reco::Track &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc );
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
