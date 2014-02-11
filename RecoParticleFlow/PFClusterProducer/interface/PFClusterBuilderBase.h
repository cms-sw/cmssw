 #ifndef __PFClusterBuilderBase_H__
#define __PFClusterBuilderBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorFactory.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderFactory.h"

#include <string>
#include <iostream>
#include <memory>

namespace edm {
  class EventSetup;
}

class PFClusterBuilderBase {
  typedef PFClusterBuilderBase PFCBB;
  typedef PFCPositionCalculatorBase PosCalc;
 public:
  PFClusterBuilderBase(const edm::ParameterSet& conf):
    _nSeeds(0), _nClustersFound(0),    
    _minFractionToKeep(conf.getParameter<double>("minFractionToKeep")),
    _algoName(conf.getParameter<std::string>("algoName")) { 
    const edm::ParameterSet& pcConf = conf.getParameterSet("positionCalc");
    const std::string& algo = pcConf.getParameter<std::string>("algoName");
    PosCalc* calcp = PFCPositionCalculatorFactory::get()->create(algo, pcConf);
    _positionCalc.reset(calcp);
    if( conf.exists("seedFinder") ) {
      const edm::ParameterSet& sfConf = conf.getParameterSet("seedFinder");
      const std::string& sfAlgo = sfConf.getParameter<std::string>("algoName");
      SeedFinderBase* sf = SeedFinderFactory::get()->create(sfAlgo, sfConf);
      _seedFinder.reset(sf);
    } else {
      _seedFinder.reset(NULL);
    }
  }
  ~PFClusterBuilderBase() { }
  // get rid of things we should never use...
  PFClusterBuilderBase(const PFCBB&) = delete;
  PFCBB& operator=(const PFCBB&) = delete;

  virtual void update(const edm::EventSetup&) { }

  virtual void buildPFClusters(const reco::PFClusterCollection& topos,
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
  std::unique_ptr<SeedFinderBase> _seedFinder;

 private:
  std::string _algoName;
  
};

#endif
