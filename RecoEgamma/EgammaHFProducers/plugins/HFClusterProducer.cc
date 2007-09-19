//Package:    EgammaHFProdcers
// Class  :    HFClusterProducer
// Original Author:  Kevin Klapoetke (minnesota)
//        
// $Id: HFClusterProducer.cc,v 1.2 2007/09/19 Kevin Klapoetke
//




#include <iostream>
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "RecoEgamma/EgammaHFProducers/plugins/HFClusterProducer.h"

HFClusterProducer::HFClusterProducer(edm::ParameterSet const& conf) {
  produces<HFEMClusterShapeCollection>();
  produces<BasicClusterCollection>();
  produces<SuperClusterCollection>();
  produces<HFEMClusterShapeAssociationCollection>(); 
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
  std::auto_ptr<HFEMClusterShapeAssociationCollection> retdata4(new HFEMClusterShapeAssociationCollection());
 
 
  algo_.clusterize(*hf_hits, *geometry, *retdata1, *retdata2, *retdata3);
  edm::OrphanHandle<SuperClusterCollection> SupHandle;
  edm::OrphanHandle<HFEMClusterShapeCollection> ShapeHandle;

  // put the results
  ShapeHandle=e.put(retdata1);
  e.put(retdata2);
  SupHandle=e.put(retdata3);
  for (unsigned int i=0; i < ShapeHandle->size();i++){
    retdata4->insert(edm::Ref<SuperClusterCollection>(SupHandle,i),edm::Ref<HFEMClusterShapeCollection>(ShapeHandle,i));
  }


  e.put(retdata4);

}
