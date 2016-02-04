#ifndef _FWPFTrack3DProxyBuilder_H_
#define _FWPFTrack3DProxyBuilder_H_

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


// System include files
#include "TEvePointSet.h"

// User include files
#include "FWPFTrackBaseProxyBuilder.h"

//-----------------------------------------------------------------------------
// FWPFTrack3DProxyBuilder
//-----------------------------------------------------------------------------
class FWPFTrack3DProxyBuilder : public FWPFTrackBaseProxyBuilder
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFTrack3DProxyBuilder(){}
      virtual ~FWPFTrack3DProxyBuilder(){}

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFTrack3DProxyBuilder( const FWPFTrack3DProxyBuilder& );
      const FWPFTrack3DProxyBuilder& operator=( const FWPFTrack3DProxyBuilder& );

   // --------------------- Member Functions --------------------------
      float        linearInterpolation( const TEveVector &p1, const TEveVector &p2, float r );
      virtual void build( const reco::Track &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc );
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
