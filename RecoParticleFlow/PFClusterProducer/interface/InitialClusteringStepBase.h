#ifndef __InitialClusteringStepBase_H__
#define __InitialClusteringStepBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

#include <string>
#include <iostream>
#include <unordered_map>

namespace edm {
  class EventSetup;
}

class InitialClusteringStepBase {
  typedef InitialClusteringStepBase ICSB;
 public:
  InitialClusteringStepBase(const edm::ParameterSet& conf):    
    _nSeeds(0), _nClustersFound(0),
    _layerMap({ {"PS2",(int)PFLayer::PS2},
	        {"PS1",(int)PFLayer::PS1},
	        {"ECAL_ENDCAP",(int)PFLayer::ECAL_ENDCAP},
	        {"ECAL_BARREL",(int)PFLayer::ECAL_BARREL},
	        {"NONE",(int)PFLayer::NONE},
	        {"HCAL_BARREL1",(int)PFLayer::HCAL_BARREL1},
	        {"HCAL_BARREL2_RING0",(int)PFLayer::HCAL_BARREL2},
		{"HCAL_BARREL2_RING1",100*(int)PFLayer::HCAL_BARREL2},
	        {"HCAL_ENDCAP",(int)PFLayer::HCAL_ENDCAP},
	        {"HF_EM",(int)PFLayer::HF_EM},
		{"HF_HAD",(int)PFLayer::HF_HAD} }),
    _algoName(conf.getParameter<std::string>("algoName")) { 
    const std::vector<edm::ParameterSet>& thresholds =
    conf.getParameterSetVector("thresholdsByDetector");
    for( const auto& pset : thresholds ) {
      const std::string& det = pset.getParameter<std::string>("detector");
      const double& thresh_E = 
	pset.getParameter<double>("gatheringThreshold");
      const double& thresh_pT = 
	pset.getParameter<double>("gatheringThresholdPt");
      const double thresh_pT2 = thresh_pT*thresh_pT;
      auto entry = _layerMap.find(det);
      if( entry == _layerMap.end() ) {
	throw cms::Exception("InvalidDetectorLayer")
	  << "Detector layer : " << det << " is not in the list of recognized"
	  << " detector layers!";
      }
      _thresholds.emplace(_layerMap.find(det)->second, 
			  std::make_pair(thresh_E,thresh_pT2));
  }
  }
  virtual ~InitialClusteringStepBase() { }
  // get rid of things we should never use...
  InitialClusteringStepBase(const ICSB&) = delete;
  ICSB& operator=(const ICSB&) = delete;

  virtual void update(const edm::EventSetup&) { }

  virtual void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
			     const std::vector<bool>& mask,  // mask flags
			     const std::vector<bool>& seeds, // seed flags
			     reco::PFClusterCollection&) = 0; //output

  std::ostream& operator<<(std::ostream& o) {
    o << "InitialClusteringStep with algo \"" << _algoName 
      << "\" located " << _nSeeds << " seeds and built " 
      << _nClustersFound << " clusters from those seeds. ";
    return o;
  }

  void reset() { _nSeeds = _nClustersFound = 0; }

 protected:
  reco::PFRecHitRef makeRefhit( const edm::Handle<reco::PFRecHitCollection>& h,
				const unsigned i ) const { 
    return reco::PFRecHitRef(h,i);
  }
  unsigned _nSeeds, _nClustersFound; // basic performance information
  const std::unordered_map<std::string,int> _layerMap;
  std::unordered_map<int,std::pair<double,double> > 
    _thresholds;
 
 private:
  const std::string _algoName;
  
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< InitialClusteringStepBase* (const edm::ParameterSet&) > InitialClusteringStepFactory;

#endif
