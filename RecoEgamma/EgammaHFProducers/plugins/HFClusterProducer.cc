#include <iostream>
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "RecoEgamma/EgammaHFProducers/plugins/HFClusterProducer.h"

HFClusterProducer::HFClusterProducer(edm::ParameterSet const& conf) {
  produces<HFEMClusterShapeCollection>();
  produces<BasicClusterCollection>();
  produces<SuperClusterCollection>();
  
}

void HFClusterProducer::produce(edm::Event & e, edm::EventSetup const& iSetup) {  
  
  edm::Handle<HFRecHitCollection> hf_hits;
  
  e.getByType(hf_hits);
  
  edm::ESHandle<CaloGeometry> geometry;
  iSetup.get<IdealGeometryRecord>().get(geometry);
  
  // create return data
  std::auto_ptr<HFEMClusterShapeCollection> retdata1(new HFEMClusterShapeCollection());
  std::auto_ptr<BasicClusterCollection> retdata2(new BasicClusterCollection());
  std::auto_ptr<SuperClusterCollection> retdata3(new SuperClusterCollection());
  
  algo_.clusterize(*hf_hits, *geometry, *retdata1, *retdata2, *retdata3);
  
  // put the results
  e.put(retdata1);
  e.put(retdata2);
  e.put(retdata3);
}
