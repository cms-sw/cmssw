#ifndef __PFClusterBuilderBase_H__
#define __PFClusterBuilderBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

#include <string>
#include <iostream>
#include <memory>

namespace edm {
  class EventSetup;
}

class PFClusterBuilderBase {
  typedef PFClusterBuilderBase PFCBB;  
 public:
  typedef PFCPositionCalculatorBase PosCalc;
  PFClusterBuilderBase(const edm::ParameterSet& conf):
    _nSeeds(0), _nClustersFound(0),    
    _minFractionToKeep(conf.getParameter<double>("minFractionToKeep")),
    _algoName(conf.getParameter<std::string>("algoName")) {
    _positionCalc.reset(NULL);
    if( conf.exists("positionCalc") ) {
      const edm::ParameterSet& pcConf = conf.getParameterSet("positionCalc");
      const std::string& algo = pcConf.getParameter<std::string>("algoName");
      PosCalc* calcp = PFCPositionCalculatorFactory::get()->create(algo, 
								   pcConf);
      _positionCalc.reset(calcp);
    }
  }
  ~PFClusterBuilderBase() { }
  // get rid of things we should never use...
  PFClusterBuilderBase(const PFCBB&) = delete;
  PFCBB& operator=(const PFCBB&) = delete;

  virtual void update(const edm::EventSetup&) { }

  virtual void buildClusters(const reco::PFClusterCollection& topos,
			     const std::vector<bool>& seedable,
			     reco::PFClusterCollection& outclus) = 0;

  std::ostream& operator<<(std::ostream& o) const {
    o << "PFClusterBuilder with algo \"" << _algoName 
      << "\" located " << _nSeeds << " seeds and built " 
      << _nClustersFound << " PFClusters from those seeds"
      << " using position calculation: " << _positionCalc->name()
      << "." << std::endl;
    return o;
  }

  void reset() { _nSeeds = _nClustersFound = 0; }

 protected:
  unsigned _nSeeds, _nClustersFound; // basic performance information
  const float _minFractionToKeep; // min fraction value to keep in clusters
  std::unique_ptr<PosCalc> _positionCalc;

 private:
  std::string _algoName;
  
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< PFClusterBuilderBase* (const edm::ParameterSet&) > PFClusterBuilderFactory;

#endif
