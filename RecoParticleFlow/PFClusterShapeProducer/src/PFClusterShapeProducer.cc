#include "RecoParticleFlow/PFClusterShapeProducer/interface/PFClusterShapeProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"


PFClusterShapeProducer::PFClusterShapeProducer(const edm::ParameterSet & ps)
{
  shapesLabel_ = ps.getParameter<std::string>("PFClusterShapesLabel");

  clustersLabel_ = ps.getParameter<std::string>("PFClustersLabel");
  clustersProducer_ = ps.getParameter<std::string>("PFClustersProducer");

  rechitsLabel_ = ps.getParameter<std::string>("PFRechitsLabel");
  rechitsProducer_ = ps.getParameter<std::string>("PFRechitsProducer");

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
  es.get<IdealGeometryRecord>().get(geoHandle);

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
  try 
    {
      evt.getByLabel(clustersProducer_, clustersLabel_, handle);
      if (!handle.isValid())
	{
	  edm::LogError("PFClusterShapeProducerError") << ("Could not get a handle on the PFClusters");
	  exit(-1);
	}
    }
  catch (cms::Exception & ex)
    {
      edm::LogError("PFClusterShapeProducerError") << ("Could not get a handle on the PFClusters");
    }

  return handle;
}

edm::Handle<reco::PFRecHitCollection>
PFClusterShapeProducer::getRecHitCollection(edm::Event & evt)
{
  edm::Handle<reco::PFRecHitCollection> handle;
  try 
    {
      evt.getByLabel(rechitsProducer_, rechitsLabel_, handle);
      if (!handle.isValid())
	{
	  edm::LogError("PFClusterShapeProducerError") << ("Could not get a handle on the PF Rechits");
	  exit(-1);
	}
    }
  catch (cms::Exception & ex)
    {
      edm::LogError("PFClusterShapeProducerError") << ("Could not get a handle on the PFClusters");
    }

  return handle;
}

