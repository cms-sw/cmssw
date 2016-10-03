#ifndef PixelTrackFitting_PixelTrackFilterByKinematics_H
#define PixelTrackFitting_PixelTrackFilterByKinematics_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterBase.h"

namespace edm {class ParameterSet; class EventSetup; }

class PixelTrackFilterByKinematics : public PixelTrackFilterBase {
public:
  PixelTrackFilterByKinematics( const edm::ParameterSet& cfg);
  PixelTrackFilterByKinematics( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);
  PixelTrackFilterByKinematics(double ptmin = 0.9, double tipmax = 0.1, double chi2max = 100.);
  virtual ~PixelTrackFilterByKinematics();
  void update(const edm::Event&, const edm::EventSetup&) override;
  virtual bool operator()(const reco::Track*, const PixelTrackFilterBase::Hits & hits) const override;
private:
  float theoPtMin, theNSigmaInvPtTolerance; 
  float theTIPMax, theNSigmaTipMaxTolerance;
  float theChi2Max;
};
#endif
