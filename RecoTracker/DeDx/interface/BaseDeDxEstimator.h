#ifndef RecoTrackerDeDx_BaseDeDxEstimator_h
#define RecoTrackerDeDx_BaseDeDxEstimator_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

class BaseDeDxEstimator
{
public: 
  virtual ~BaseDeDxEstimator() {}
  virtual std::pair<float,float> dedx(const reco::DeDxHitCollection& Hits) = 0;
  virtual void beginRun(edm::Run const& run, const edm::EventSetup& iSetup) {}
};

#endif

