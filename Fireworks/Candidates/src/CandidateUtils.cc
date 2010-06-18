#include "Fireworks/Candidates/interface/CandidateUtils.h"
#include "TEveVSDStructs.h"
#include "TEveTrack.h"
#include "TEveVector.h"
#include "TEveStraightLineSet.h"

namespace fireworks {

TEveTrack*
prepareCandidate(const reco::Candidate& track,
                 TEveTrackPropagator* propagator)
{
   TEveRecTrack t;
   t.fBeta = 1.;
   t.fP = TEveVector( track.px(), track.py(), track.pz() );
   t.fV = TEveVector( track.vertex().x(), track.vertex().y(), track.vertex().z() );
   t.fSign = track.charge();
   TEveTrack* trk = new TEveTrack(&t, propagator);
   return trk;
}
  
void
addStraightLineSegment( TEveStraightLineSet* marker,
                        reco::Candidate const* cand,
                        double scale_factor)
{
   double phi = cand->phi();
   double theta = cand->theta();
   double size = cand->pt() * scale_factor;
   marker->AddLine( 0, 0, 0, size * cos(phi)*sin(theta), size *sin(phi)*sin(theta), size*cos(theta));
}
}
