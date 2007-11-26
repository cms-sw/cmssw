#ifndef PixelTrackFitting_PixelTrackFilterByKinematics_H
#define PixelTrackFitting_PixelTrackFilterByKinematics_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
namespace edm {class ParameterSet;}

class PixelTrackFilterByKinematics : public PixelTrackFilter {
public:
  PixelTrackFilterByKinematics( const edm::ParameterSet& cfg);
  PixelTrackFilterByKinematics(double ptmin = 0.9, double tipmax = 0.1, double chi2max = 100.);
  virtual ~PixelTrackFilterByKinematics();
  virtual bool operator()(const reco::Track*) const;
private:
  double thePtMin, theNSigmaInvPtTolerance; 
  double theTIPMax, theNSigmaTipMaxTolerance;
  double theChi2Max;
};
#endif
