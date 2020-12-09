#include "L1Trigger/VertexFinder/interface/RecoVertex.h"

namespace l1tVertexFinder {

  void RecoVertex::computeParameters(unsigned int weightedmean, double highPtThreshold, int highPtBehavior) {
    pT_ = 0.;
    z0_ = 0.;
    highPt_ = false;
    highestPt_ = 0.;
    numHighPtTracks_ = 0;

    float SumZ = 0.;
    float z0square = 0.;
    float trackPt = 0.;

    for (const L1Track* track : tracks_) {
      trackPt = track->pt();
      if (trackPt > highPtThreshold) {
        highPt_ = true;
        numHighPtTracks_++;
        highestPt_ = (trackPt > highestPt_) ? trackPt : highestPt_;
        if (highPtBehavior == 0) continue; // ignore this track
        else if (highPtBehavior == 1) trackPt = highPtThreshold;  // saturate
      }

      pT_ += trackPt;
      SumZ += track->z0() * std::pow(trackPt, weightedmean);
      z0square += track->z0() * track->z0();
    }

    z0_ = SumZ / ((weightedmean > 0) ? pT_ : tracks_.size());
    z0square /= tracks_.size();
    z0width_ = sqrt(std::abs(z0_ * z0_ - z0square));
  }

}  // namespace l1tVertexFinder
