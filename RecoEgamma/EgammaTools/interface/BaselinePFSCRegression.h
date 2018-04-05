#ifndef __BASELINEPFSCREGRESSION_H__
#define __BASELINEPFSCREGRESSION_H__

#include "RecoEgamma/EgammaTools/interface/SCRegressionCalculator.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h" 
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <vector>

class BaselinePFSCRegression {
 public:
  BaselinePFSCRegression() : topo_record(nullptr), geom_record(nullptr) {};
  void update(const edm::EventSetup&);
  void set(const reco::SuperCluster&, std::vector<float>&) const;
  void setTokens(const edm::ParameterSet&, edm::ConsumesCollector&&);
  void setEvent(const edm::Event&);

 private:
  const CaloTopologyRecord* topo_record;
  const CaloGeometryRecord* geom_record;
  EcalClusterLocal ecl_;
  edm::ESHandle<CaloTopology> calotopo;
  edm::ESHandle<CaloGeometry> calogeom;
  edm::EDGetTokenT<EcalRecHitCollection>          inputTagEBRecHits_;
  edm::EDGetTokenT<EcalRecHitCollection>          inputTagEERecHits_;
  edm::EDGetTokenT<reco::VertexCollection>        inputTagVertices_;
  edm::Handle<reco::VertexCollection> vertices;
  edm::Handle<EcalRecHitCollection>  rechitsEB,rechitsEE;
};

typedef SCRegressionCalculator<BaselinePFSCRegression> PFSCRegressionCalc;

#endif
