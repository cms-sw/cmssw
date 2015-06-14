#ifndef __SimplePositionCalc_H__
#define __SimplePositionCalc_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "RecoParticleFlow/PFClusterProducer/interface/ECALRecHitResolutionProvider.h"

class SimplePositionCalc : public PFCPositionCalculatorBase {
 public:
  SimplePositionCalc(const edm::ParameterSet& conf) :
    PFCPositionCalculatorBase(conf) {  }
  SimplePositionCalc(const SimplePositionCalc&) = delete;
  SimplePositionCalc& operator=(const SimplePositionCalc&) = delete;

  void calculateAndSetPosition(reco::PFCluster&);
  void calculateAndSetPositions(reco::PFClusterCollection&);

 private:
  void calculateAndSetPositionActual(reco::PFCluster&) const;
};

DEFINE_EDM_PLUGIN(PFCPositionCalculatorFactory,
		  SimplePositionCalc,
		  "SimplePositionCalc");

#endif
