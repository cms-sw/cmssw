#ifndef __TopoClusterBuilderBase_H__
#define __TopoClusterBuilderBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include <string>
#include <iostream>

class TopoClusterBuilderBase {
  typedef TopoClusterBuilderBase TCBB;
 public:
  TopoClusterBuilderBase(const edm::ParameterSet& conf):    
    _nSeeds(0), _nClustersFound(0),
    _seedingThreshold(conf.getParameter<float>("seedingThreshold")),
    _gatheringThreshold(conf.getParameter<float>("gatheringThreshold")),
    _algoName(conf.getParameter<std::string>("algoName")) { }
  ~TopoClusterBuilderBase() { }
  // get rid of things we should never use...
  TopoClusterBuilderBase(const TCBB&) = delete;
  TCBB& operator=(const TCBB&) = delete;

  virtual void buildTopoClusters(const reco::PFRecHitCollection&,
				 reco::PFClusterCollection&) = 0;

  std::ostream& operator<<(std::ostream& o) {
    o << "TopoClusterBuilder with algo \"" << _algoName 
      << "\" located " << _nSeeds << " seeds and built " 
      << _nClustersFound << " TopoClusters from those seeds." << std::endl;
    return o;
  }

  void reset() { _nSeeds = _nClustersFound = 0; }

 protected:
  unsigned _nSeeds, _nClustersFound; // basic performance information
  float _seedingThreshold;
  float _gatheringThreshold; // RecHit energy threshold to keep going

 private:
  std::string _algoName;
  
};

#endif
