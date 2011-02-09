#ifndef _FWPFTRACKUTILS_H_
#define _FWPFTRACKUTILS_h_

// System include files
#include "TEveTrack.h"

// User include files
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "DataFormats/TrackReco/interface/Track.h"

//-----------------------------------------------------------------------------
// FWPFTrackUtils
//-----------------------------------------------------------------------------

class FWPFTrackUtils
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFTrackUtils();
      virtual ~FWPFTrackUtils(){}

   // ------------------------ Functionality --------------------------
      TEveTrack               *getTrack( const reco::Track &iData );
      TEveVector              lineCircleIntersect( const TEveVector &v1, const TEveVector &v2, float r );
      TEveVector              lineLineIntersect( const TEveVector &v1, const TEveVector &v2,
                                                 const TEveVector &v3, const TEveVector &v4 );
      TEveVector              cross( const TEveVector &v1, const TEveVector &v2 );
      float                   linearInterpolation( const TEveVector &p1, const TEveVector &p2, float r );
      float                   dot( const TEveVector &v1, const TEveVector &v2 );
      float                   sgn( float val );
      bool                    checkIntersect( const TEveVector &vec, float r );

   // ---------------------- Accessor Methods ---------------------------
      float                   getCaloR1() { return m_caloR1; }
      float                   getCaloR2() { return m_caloR2; }
		float							getCaloR3() { return m_caloR3; }
      float                   getCaloZ1() { return m_caloZ1; }
      float                   getCaloZ2() { return m_caloZ2; }

   private:
      FWPFTrackUtils( const FWPFTrackUtils& );
      const FWPFTrackUtils& operator=( const FWPFTrackUtils& );

   // ----------------------- Data Members ---------------------------
      TEveTrackPropagator     *m_trackerTrackPropagator;
      TEveTrackPropagator     *m_trackPropagator;
      FWMagField              *m_magField;

      float                   m_caloR1;
      float                   m_caloR2;
		float							m_caloR3;
      float                   m_caloZ1;
      float                   m_caloZ2;
};
#endif
