#include "RecoParticleFlow/PFClusterShapeProducer/plugins/PFClusterShapeProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <sstream>

using namespace edm;
using namespace std;

PFClusterShapeProducer::PFClusterShapeProducer(const edm::ParameterSet & ps)
{
  shapesLabel_ = ps.getParameter<std::string>("PFClusterShapesLabel");

  inputTagPFClustersECAL_ 
    = ps.getParameter<InputTag>("PFClustersECAL");
  inputTagPFRecHitsECAL_ 
    = ps.getParameter<InputTag>("PFRecHitsECAL");
  
  csAlgo_p = new PFClusterShapeAlgo(ps.getParameter<bool>("useFractions"),
				    ps.getParameter<double>("W0"));

  produces<reco::ClusterShapeCollection>(shapesLabel_);
  produces<reco::PFClusterShapeAssociationCollection>(shapesLabel_);
}


PFClusterShapeProducer::~PFClusterShapeProducer()
{
  delete csAlgo_p;
}


void PFClusterShapeProducer::produce(edm::Event & evt, const edm::EventSetup & es)
{

  edm::Handle<reco::PFClusterCollection> 
    clusterHandle = getClusterCollection(evt);
  edm::Handle<reco::PFRecHitCollection>
    rechitHandle = getRecHitCollection(evt);

  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);

  const CaloSubdetectorGeometry * barrelGeo_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  const CaloSubdetectorTopology * barrelTop_p = new EcalBarrelTopology(geoHandle);
  const CaloSubdetectorGeometry * endcapGeo_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  const CaloSubdetectorTopology * endcapTop_p = new EcalEndcapTopology(geoHandle);

  std::auto_ptr<reco::ClusterShapeCollection> 
    csCollection_ap(csAlgo_p->makeClusterShapes(clusterHandle, rechitHandle, 
						barrelGeo_p, barrelTop_p,
						endcapGeo_p, endcapTop_p));
  
  edm::OrphanHandle<reco::ClusterShapeCollection> shape_h = evt.put(csCollection_ap, shapesLabel_);
  
  std::auto_ptr<reco::PFClusterShapeAssociationCollection> association_ap(new reco::PFClusterShapeAssociationCollection);
 
  for (unsigned int i = 0; i < clusterHandle->size(); i++){
    association_ap->insert(edm::Ref<reco::PFClusterCollection>(clusterHandle, i), 
			   edm::Ref<reco::ClusterShapeCollection>(shape_h, i));
  } 
  
  evt.put(association_ap, shapesLabel_);

  delete barrelTop_p;
  delete endcapTop_p;
}


edm::Handle<reco::PFClusterCollection>
PFClusterShapeProducer::getClusterCollection(edm::Event & evt)
{
  edm::Handle<reco::PFClusterCollection> handle;
  
  bool found = evt.getByLabel(inputTagPFClustersECAL_, handle);
  if (!found) {
    ostringstream err;
    err<<"cannot find clusters: "<<inputTagPFClustersECAL_;
    LogError("PFSimParticleProducer")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }

  return handle;
}



edm::Handle<reco::PFRecHitCollection>
PFClusterShapeProducer::getRecHitCollection(edm::Event & evt)
{
  edm::Handle<reco::PFRecHitCollection> handle;

  bool found = evt.getByLabel(inputTagPFRecHitsECAL_, handle);
  if (!found) {
    ostringstream err;
    err<<"cannot find rechits: "<<inputTagPFRecHitsECAL_;
    LogError("PFSimParticleProducer")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }

  return handle;
}

