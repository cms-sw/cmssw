#ifndef _FWPFTRACKRPZLEGOPROXYBUILDER_H_
#define _FWPFTRACKRPZLEGOPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFTrackRPZProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/ParticleFlow/interface/FWPFTrackUtils.h"

//-----------------------------------------------------------------------------
// FWPFTrackRPZProxyBuilder
//-----------------------------------------------------------------------------
class FWPFTrackRPZProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track>
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFTrackRPZProxyBuilder(){}
      virtual ~FWPFTrackRPZProxyBuilder(){}

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFTrackRPZProxyBuilder( const FWPFTrackRPZProxyBuilder& );
      const FWPFTrackRPZProxyBuilder& operator=( const FWPFTrackRPZProxyBuilder& );

   // --------------------- Member Functions --------------------------
      virtual void build( const reco::Track &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc );
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
