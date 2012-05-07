#ifndef _FWPFTRACKUTILS_H_
#define _FWPFTRACKUTILS_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFTrackUtils
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//       Created:    16/02/2011
//

// System include files
#include "TEveTrack.h"
#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"

// User include files
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/ParticleFlow/interface/FWPFUtils.h"

#include "DataFormats/TrackReco/interface/Track.h"

//-----------------------------------------------------------------------------
// FWPFTrackUtils
//-----------------------------------------------------------------------------
class FWPFTrackUtils
{
   public:
      enum Type { LEGO=0, RPZ=1 };

   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFTrackUtils(){ m_trackUtils = new FWPFUtils(); m_trackUtils->initPropagator(); }
      virtual ~FWPFTrackUtils(){ delete m_trackUtils; }

   // --------------------- Member Functions --------------------------
      TEveStraightLineSet  *setupLegoTrack( const reco::Track& );
      TEveTrack            *setupRPZTrack( const reco::Track& );
      TEvePointSet         *getCollisionMarkers( const TEveTrack* );

   private:
      FWPFTrackUtils( const FWPFTrackUtils& );                    // Stop default copy constructor
      const FWPFTrackUtils& operator=( const FWPFTrackUtils& );   // Stop default assignment operator

   // ----------------------- Data Members ----------------------------
      FWPFUtils *m_trackUtils;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
