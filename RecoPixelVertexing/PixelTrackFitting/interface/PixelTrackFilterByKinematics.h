#ifndef PixelTrackFitting_PixelTrackFilterByKinematics_H
#define PixelTrackFitting_PixelTrackFilterByKinematics_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"

namespace edm {class ParameterSet; class EventSetup; }

class PixelTrackFilterByKinematics : public PixelTrackFilter {
public:
  PixelTrackFilterByKinematics( const edm::ParameterSet& cfg);
  PixelTrackFilterByKinematics( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);
  PixelTrackFilterByKinematics(double ptmin = 0.9, double tipmax = 0.1, double chi2max = 100.);
  virtual ~PixelTrackFilterByKinematics();
  void update(const edm::Event&, const edm::EventSetup&) override;
  virtual bool operator()(const reco::Track*) const;
  virtual bool operator()(const reco::Track*, const PixelTrackFilter::Hits & hits) const;
private:
  float theoPtMin, theNSigmaInvPtTolerance; 
  float theTIPMax, theNSigmaTipMaxTolerance;
  float theChi2Max;
};
#endif
