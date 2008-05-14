//Package:    EgammaHFProdcers
// Class  :    HFEMClusterProducer
// Original Author:  Kevin Klapoetke (minnesota)
//        
// $Id: HFEMClusterProducer.cc,v 1.2 2007/09/19 Kevin Klapoetke
//

#include <iostream>
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEgamma/EgammaHFProducers/plugins/HFEMClusterProducer.h"
using namespace reco;
HFEMClusterProducer::HFEMClusterProducer(edm::ParameterSet const& conf): hfreco_(conf.getUntrackedParameter<edm::InputTag>("hits")) {
  produces<reco::HFEMClusterShapeCollection>();
  produces<reco::BasicClusterCollection>();
  produces<reco::SuperClusterCollection>();
  produces<reco::HFEMClusterShapeAssociationCollection>(); 
  algo_.setup(conf.getParameter<double>("minTowerEnergy"));
}

void HFEMClusterProducer::produce(edm::Event & e, edm::EventSetup const& iSetup) {  
  
  edm::Handle<HFRecHitCollection> hf_hits;
  
  e.getByLabel(hfreco_,hf_hits);
  
  edm::ESHandle<CaloGeometry> geometry;
  iSetup.get<CaloGeometryRecord>().get(geometry);
  
  // create return data
  std::auto_ptr<reco::HFEMClusterShapeCollection> retdata1(new HFEMClusterShapeCollection());
  std::auto_ptr<reco::BasicClusterCollection> retdata2(new BasicClusterCollection());
  std::auto_ptr<reco::SuperClusterCollection> retdata3(new SuperClusterCollection());
  std::auto_ptr<reco::HFEMClusterShapeAssociationCollection> retdata4(new HFEMClusterShapeAssociationCollection());
 
 
  algo_.clusterize(*hf_hits, *geometry, *retdata1, *retdata2, *retdata3);
  edm::OrphanHandle<reco::SuperClusterCollection> SupHandle;
  edm::OrphanHandle<reco::HFEMClusterShapeCollection> ShapeHandle;

  // put the results
  ShapeHandle=e.put(retdata1);
  e.put(retdata2);
  SupHandle=e.put(retdata3);
  for (unsigned int i=0; i < ShapeHandle->size();i++){
    retdata4->insert(edm::Ref<reco::SuperClusterCollection>(SupHandle,i),edm::Ref<reco::HFEMClusterShapeCollection>(ShapeHandle,i));
  }


  e.put(retdata4);

}
