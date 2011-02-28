#ifndef _FWPFUTILS_H_
#define _FWPFUTILS_h_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFUtils
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include "TEveTrack.h"

// User include files
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "DataFormats/TrackReco/interface/Track.h"

//-----------------------------------------------------------------------------
// FWPFUtils
//-----------------------------------------------------------------------------
class FWPFUtils
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFUtils();
      virtual ~FWPFUtils(){}

   // --------------------- Member Functions --------------------------
      TEveTrack               *getTrack( const reco::Track &iData );
      TEveVector              lineCircleIntersect( const TEveVector &v1, const TEveVector &v2, float r );
      TEveVector              lineLineIntersect( const TEveVector &v1, const TEveVector &v2,
                                                 const TEveVector &v3, const TEveVector &v4 );
      TEveVector              cross( const TEveVector &v1, const TEveVector &v2 );
      float                   linearInterpolation( const TEveVector &p1, const TEveVector &p2, float r );
      float                   dot( const TEveVector &v1, const TEveVector &v2 );
      float                   sgn( float val );
      bool                    checkIntersect( const TEveVector &vec, float r );
      void                    initPropagator();

   // --------------------- Accessor Methods --------------------------
      float                   getCaloR1() { return m_caloR1;   }
      float                   getCaloR2() { return m_caloR2;   }
      float                   getCaloR3() { return m_caloR3;   }
      float                   getCaloZ1() { return m_caloZ1;   }
      float                   getCaloZ2() { return m_caloZ2;   }
      FWMagField              *getField() { return m_magField; }

   private:
      FWPFUtils( const FWPFUtils& );
      const FWPFUtils& operator=( const FWPFUtils& );

   // ----------------------- Data Members ----------------------------
      TEveTrackPropagator     *m_trackerTrackPropagator;
      TEveTrackPropagator     *m_trackPropagator;
      FWMagField              *m_magField;

      float                   m_caloR1;
      float                   m_caloR2;
      float                   m_caloR3;
      float                   m_caloZ1;
      float                   m_caloZ2;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
