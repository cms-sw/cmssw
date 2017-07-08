#ifndef _FWPFTRACKUTILS_H_
#define _FWPFTRACKUTILS_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFTrackSingleton, FWPFTrackUtils
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
#include "Fireworks/ParticleFlow/interface/FWPFGeom.h"
#include "Fireworks/ParticleFlow/interface/FWPFMaths.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "DataFormats/TrackReco/interface/Track.h"

//-----------------------------------------------------------------------------
// FWPFTrackSingleton
//-----------------------------------------------------------------------------
/* Created as singleton because only 1 instance of propagators and magfield should be shared
 * between track proxybuilder classes */
class FWPFTrackSingleton
{
   public:
   // --------------------- Member Functions --------------------------
      static FWPFTrackSingleton *Instance();

      inline TEveTrackPropagator *getTrackerTrackPropagator()  { return m_trackerTrackPropagator;  }
      inline TEveTrackPropagator *getTrackPropagator()         { return m_trackPropagator;         }
      inline FWMagField          *getField()                   { return m_magField;                }

   protected:
      FWPFTrackSingleton( const FWPFTrackSingleton& );                     // Stop default copy constructor
      const FWPFTrackSingleton& operator=( const FWPFTrackSingleton& );    // Stop default assignment operator

   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFTrackSingleton(){ initPropagator(); }
      virtual ~FWPFTrackSingleton(){ instanceFlag = false; }

   private:
   // --------------------- Member Functions --------------------------
      void initPropagator();

   // ----------------------- Data Members ----------------------------
      static FWPFTrackSingleton  *pInstance; // Pointer to instance if one exists
      static bool instanceFlag;

      TEveTrackPropagator       *m_trackerTrackPropagator;
      TEveTrackPropagator       *m_trackPropagator;
      FWMagField                *m_magField;
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_


//-----------------------------------------------------------------------------
// FWPFTrackUtils
//-----------------------------------------------------------------------------
class FWPFTrackUtils
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFTrackUtils();
      virtual ~FWPFTrackUtils(){}

   // --------------------- Member Functions --------------------------
      TEveStraightLineSet  *setupLegoTrack( const reco::Track& );
      TEveTrack            *setupTrack( const reco::Track& );
      TEvePointSet         *getCollisionMarkers( const TEveTrack* );

   private:
      FWPFTrackUtils( const FWPFTrackUtils& ) = delete;                    // Stop default copy constructor
      const FWPFTrackUtils& operator=( const FWPFTrackUtils& ) = delete;   // Stop default assignment operator

      TEveTrack            *getTrack( const reco::Track& );

      FWPFTrackSingleton *m_singleton;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
