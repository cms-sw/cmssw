#ifndef _FWPFTRACKRPZPROXYBUILDER_H_
#define _FWPFTRACKRPZPROXYBUILDER_H_

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
#include "FWPFTrackBaseProxyBuilder.h"

//-----------------------------------------------------------------------------
// FWPFTrackRPZProxyBuilder
//-----------------------------------------------------------------------------
class FWPFTrackRPZProxyBuilder : public FWPFTrackBaseProxyBuilder
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
