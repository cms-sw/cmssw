#ifndef PixelTrackFitting_PixelTrackFilterByKinematics_H
#define PixelTrackFitting_PixelTrackFilterByKinematics_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
namespace edm {class ParameterSet;}

class PixelTrackFilterByKinematics : public PixelTrackFilter {
public:
  PixelTrackFilterByKinematics( const edm::ParameterSet& cfg);
  PixelTrackFilterByKinematics(float ptmin = 0.9, float tipmax = 0.1, float chi2max = 100.);
  virtual ~PixelTrackFilterByKinematics();
  virtual bool operator()(const reco::Track*) const;
private:
  float thePtMin;
  float theTIPMax;
  float theChi2Max;
};
#endif
