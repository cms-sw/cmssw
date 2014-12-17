#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitNavigatorBase_h
#define RecoParticleFlow_PFClusterProducer_PFRecHitNavigatorBase_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <unordered_map>

class PFRecHitNavigatorBase {
 public:
  typedef std::unordered_map<unsigned,unsigned> DetIdToHitIdx;

  PFRecHitNavigatorBase() {}
  PFRecHitNavigatorBase(const edm::ParameterSet& iConfig) {}

  virtual ~PFRecHitNavigatorBase() {}

  virtual void beginEvent(const edm::EventSetup&)=0;
  virtual void associateNeighbours(reco::PFRecHit&,std::auto_ptr<reco::PFRecHitCollection>&,edm::RefProd<reco::PFRecHitCollection>&)=0;
  virtual void associateNeighbours(reco::PFRecHit&,
				   std::auto_ptr<reco::PFRecHitCollection>&,
				   const DetIdToHitIdx&,
				   edm::RefProd<reco::PFRecHitCollection>&) {};


 protected:
  // assumes sorted
  void associateNeighbour(const DetId& id, reco::PFRecHit& hit,std::auto_ptr<reco::PFRecHitCollection>& hits,edm::RefProd<reco::PFRecHitCollection>& refProd,short eta, short phi,short depth) {
    const reco::PFRecHit temp(id,PFLayer::NONE,0.0,math::XYZPoint(0,0,0),math::XYZVector(0,0,0),std::vector<math::XYZPoint>());
    auto found_hit = std::lower_bound(hits->begin(),hits->end(),
				      temp,
				      [](const reco::PFRecHit& a, 
					 const reco::PFRecHit& b){
					return a.detId() < b.detId();
				      });
    if( found_hit != hits->end() && found_hit->detId() == id.rawId() ) {
      hit.addNeighbour(eta,phi,depth,reco::PFRecHitRef(refProd,std::distance(hits->begin(),found_hit)));
    }    
  }
  // map to indices is provided
  void associateNeighbour(const DetId& id, 
			  reco::PFRecHit& hit,
			  std::auto_ptr<reco::PFRecHitCollection>& hits,
			  const DetIdToHitIdx& hitmap,
			  edm::RefProd<reco::PFRecHitCollection>& refProd,
			  short eta, short phi,short depth) {
    auto found_hit = hitmap.find(id.rawId());
    if( found_hit != hitmap.end() ) {      
      hit.addNeighbour(eta,phi,depth,reco::PFRecHitRef(refProd,found_hit->second));
    }    
  }

};



#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<PFRecHitNavigatorBase*(const edm::ParameterSet&)> PFRecHitNavigationFactory;

#endif
