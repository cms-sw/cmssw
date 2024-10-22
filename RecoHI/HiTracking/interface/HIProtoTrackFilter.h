#ifndef _HIProtoTrackFilter_h_
#define _HIProtoTrackFilter_h_

#include "RecoTracker/PixelTrackFitting/interface/PixelTrackFilterBase.h"

namespace reco {
  class BeamSpot;
}

class HIProtoTrackFilter : public PixelTrackFilterBase {
public:
  HIProtoTrackFilter(const reco::BeamSpot *beamSpot, double tipMax, double chi2Max, double ptMin);
  ~HIProtoTrackFilter() override;
  bool operator()(const reco::Track *, const PixelTrackFilterBase::Hits &hits) const override;

private:
  double theTIPMax;
  double theChi2Max, thePtMin;
  const reco::BeamSpot *theBeamSpot;
};

#endif
