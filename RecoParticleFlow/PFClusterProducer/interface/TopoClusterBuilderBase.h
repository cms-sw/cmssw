#ifndef __TopoClusterBuilderBase_H__
#define __TopoClusterBuilderBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderFactory.h"

#include <string>
#include <iostream>

namespace edm {
  class EventSetup;
}

class TopoClusterBuilderBase {
  typedef TopoClusterBuilderBase TCBB;
 public:
  TopoClusterBuilderBase(const edm::ParameterSet& conf):    
    _nSeeds(0), _nClustersFound(0),
    _gatheringThreshold(conf.getParameter<double>("gatheringThreshold")),
    _gatheringThresholdPt2(std::pow(conf.getParameter<double>("gatheringThresholdPt"),2.0)),
    _algoName(conf.getParameter<std::string>("algoName")) { }
  virtual ~TopoClusterBuilderBase() { }
  // get rid of things we should never use...
  TopoClusterBuilderBase(const TCBB&) = delete;
  TCBB& operator=(const TCBB&) = delete;

  virtual void update(const edm::EventSetup&) { }

  virtual void buildTopoClusters(const reco::PFRecHitRefVector&,//input rechits
				 const std::vector<bool>& mask,  // mask flags
				 const std::vector<bool>& seeds, // seed flags
				 reco::PFClusterCollection&) = 0; //output

  std::ostream& operator<<(std::ostream& o) {
    o << "TopoClusterBuilder with algo \"" << _algoName 
      << "\" located " << _nSeeds << " seeds and built " 
      << _nClustersFound << " TopoClusters from those seeds. ";
    return o;
  }

  void reset() { _nSeeds = _nClustersFound = 0; }

 protected:
  unsigned _nSeeds, _nClustersFound; // basic performance information
  const float _gatheringThreshold; // RecHit energy threshold to keep going
  const float _gatheringThresholdPt2; // RecHit pt^2 threshold to keep going  
 
 private:
  const std::string _algoName;
  
};

#endif
