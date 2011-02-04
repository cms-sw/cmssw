#ifndef _FPWFTRACKBASEPROXYBUILDER_H_
#define _FWPFTRACKBASEPROXYBUILDER_H_

//
// Package:             Particle Flow
// Class:               FWPFTrackBaseProxyBuilder
// Original Author:     Simon Harris
//

// System include files
#include "TEveTrack.h"
#include "TEvePointSet.h"

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/ParticleFlow/interface/FWPFTrackUtils.h"

#include "DataFormats/TrackReco/interface/Track.h"

//-----------------------------------------------------------------------------
// FWPFTrackBaseProxyBuilder
//-----------------------------------------------------------------------------

class FWPFTrackBaseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track>
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFTrackBaseProxyBuilder(){ m_trackUtils = new FWPFTrackUtils(); }
      virtual ~FWPFTrackBaseProxyBuilder(){ delete m_trackUtils; }

      REGISTER_PROXYBUILDER_METHODS();

   protected:
   // ----------------------- Data Members ---------------------------
      FWPFTrackUtils *m_trackUtils;

   private:
      FWPFTrackBaseProxyBuilder( const FWPFTrackBaseProxyBuilder& );
      const FWPFTrackBaseProxyBuilder& operator=( const FWPFTrackBaseProxyBuilder& );
};
#endif

//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
