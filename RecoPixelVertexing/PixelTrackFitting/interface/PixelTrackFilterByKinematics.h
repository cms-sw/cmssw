#ifndef PixelTrackFitting_PixelTrackFilterByKinematics_H
#define PixelTrackFitting_PixelTrackFilterByKinematics_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterBase.h"

namespace edm {class ParameterSet; class EventSetup; }

class PixelTrackFilterByKinematics : public PixelTrackFilterBase {
public:
  PixelTrackFilterByKinematics(double ptmin = 0.9, double tipmax = 0.1, double chi2max = 100.);
  PixelTrackFilterByKinematics(float optmin, float invPtTolerance, float tipmax, float tipmaxTolerance, float chi2max);
  virtual ~PixelTrackFilterByKinematics();
  virtual bool operator()(const reco::Track*, const PixelTrackFilterBase::Hits & hits) const override;
private:
  float theoPtMin, theNSigmaInvPtTolerance; 
  float theTIPMax, theNSigmaTipMaxTolerance;
  float theChi2Max;
};
#endif
