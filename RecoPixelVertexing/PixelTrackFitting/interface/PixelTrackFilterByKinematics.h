#ifndef PixelTrackFitting_PixelTrackFilterByKinematics_H
#define PixelTrackFitting_PixelTrackFilterByKinematics_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"

class PixelTrackFilterByKinematics : public PixelTrackFilter {
public:
  PixelTrackFilterByKinematics(float ptmin = 0.9, float tipmax = 0.1, float chi2max = 100.);
  virtual ~PixelTrackFilterByKinematics();
  virtual bool operator()(const reco::Track*) const;
private:
  float thePtMin;
  float theTIPMax;
  float theChi2Max;
};
#endif
