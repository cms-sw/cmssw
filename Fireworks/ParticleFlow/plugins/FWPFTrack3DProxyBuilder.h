#ifndef _FWPFTRACK3DPROXYBUILDER_H_
#define _FWPFTRACK3DPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFTrack3DProxyBuilder
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
// FWPFTrack3DProxyBuilder
//-----------------------------------------------------------------------------
class FWPFTrack3DProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track>
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFTrack3DProxyBuilder(){}
      virtual ~FWPFTrack3DProxyBuilder(){}

      using FWSimpleProxyBuilderTemplate<reco::Track>::build;
      virtual void build( const reco::Track &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc );
      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFTrack3DProxyBuilder( const FWPFTrack3DProxyBuilder& );
      const FWPFTrack3DProxyBuilder& operator=( const FWPFTrack3DProxyBuilder& );

   // --------------------- Member Functions --------------------------
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
