#ifndef __PFCPositionCalculatorBase_H__
#define __PFCPositionCalculatorBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include <string>

class PFCPositionCalculatorBase {
  typedef PFCPositionCalculatorBase PosCalc;
 public:
  PFCPositionCalculatorBase(const edm::ParameterSet& conf) :
    _minFractionInCalc(conf.getParameter<double>("minFractionInCalc")),
    _algoName(conf.getParameter<std::string>("algoName")) { }
  ~PFCPositionCalculatorBase() { }
  //get rid of things we should never use
  PFCPositionCalculatorBase(const PosCalc&) = delete;
  PosCalc& operator=(const PosCalc&) = delete;

  // here we transform one PFCluster to use the new position calculation
  virtual void calculateAndSetPosition(reco::PFCluster&) = 0;
  // here you call a loop inside to transform the whole vector
  virtual void calculateAndSetPositions(reco::PFClusterCollection&) = 0;

  const std::string& name() const { return _algoName; }
  
 protected:  
  const float _minFractionInCalc;

 private:  
  const std::string _algoName;

};

#endif
