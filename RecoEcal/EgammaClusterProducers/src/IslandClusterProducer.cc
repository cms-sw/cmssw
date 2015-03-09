// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"

// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/IslandClusterProducer.h"


IslandClusterProducer::IslandClusterProducer(const edm::ParameterSet& ps)
{
  // The verbosity level
  std::string verbosityString = ps.getParameter<std::string>("VerbosityLevel");
  if      (verbosityString == "DEBUG")   verbosity = IslandClusterAlgo::pDEBUG;
  else if (verbosityString == "WARNING") verbosity = IslandClusterAlgo::pWARNING;
  else if (verbosityString == "INFO")    verbosity = IslandClusterAlgo::pINFO;
  else                                   verbosity = IslandClusterAlgo::pERROR;

  // Parameters to identify the hit collections
  barrelRecHits_   = 
	  consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("barrelHits"));
  endcapRecHits_   = 
	  consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("endcapHits"));

  // The names of the produced cluster collections
  barrelClusterCollection_  = ps.getParameter<std::string>("barrelClusterCollection");
  endcapClusterCollection_  = ps.getParameter<std::string>("endcapClusterCollection");

  // Island algorithm parameters
  double barrelSeedThreshold = ps.getParameter<double>("IslandBarrelSeedThr");
  double endcapSeedThreshold = ps.getParameter<double>("IslandEndcapSeedThr");

  // Parameters for the position calculation:
   edm::ParameterSet posCalcParameters = 
    ps.getParameter<edm::ParameterSet>("posCalcParameters");
  posCalculator_ = PositionCalc(posCalcParameters);
  shapeAlgo_ = ClusterShapeAlgo(posCalcParameters);

  clustershapecollectionEB_ = ps.getParameter<std::string>("clustershapecollectionEB");
  clustershapecollectionEE_ = ps.getParameter<std::string>("clustershapecollectionEE");

  //AssociationMap
  barrelClusterShapeAssociation_ = ps.getParameter<std::string>("barrelShapeAssociation");
  endcapClusterShapeAssociation_ = ps.getParameter<std::string>("endcapShapeAssociation");

  // Produces a collection of barrel and a collection of endcap clusters

  produces< reco::ClusterShapeCollection>(clustershapecollectionEE_);
  produces< reco::BasicClusterCollection >(endcapClusterCollection_);
  produces< reco::ClusterShapeCollection>(clustershapecollectionEB_);
  produces< reco::BasicClusterCollection >(barrelClusterCollection_);
  produces< reco::BasicClusterShapeAssociationCollection >(barrelClusterShapeAssociation_);
  produces< reco::BasicClusterShapeAssociationCollection >(endcapClusterShapeAssociation_);

  island_p = new IslandClusterAlgo(barrelSeedThreshold, endcapSeedThreshold, posCalculator_,verbosity);

  nEvt_ = 0;
}


IslandClusterProducer::~IslandClusterProducer()
{
  delete island_p;
}


void IslandClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  clusterizeECALPart(evt, es, endcapRecHits_, endcapClusterCollection_, endcapClusterShapeAssociation_, IslandClusterAlgo::endcap); 
  clusterizeECALPart(evt, es, barrelRecHits_, barrelClusterCollection_, barrelClusterShapeAssociation_, IslandClusterAlgo::barrel);
  nEvt_++;
}


const EcalRecHitCollection * IslandClusterProducer::getCollection(edm::Event& evt,const edm::EDGetTokenT<EcalRecHitCollection>& token)
{
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByToken(token, rhcHandle);
  return rhcHandle.product();
  
}

void IslandClusterProducer::clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,const edm::EDGetTokenT<EcalRecHitCollection>& token,                                              const std::string& clusterCollection,
					       const std::string& clusterShapeAssociation,
                                               const IslandClusterAlgo::EcalPart& ecalPart)
{
  // get the hit collection from the event:
  const EcalRecHitCollection *hitCollection_p = getCollection(evt,token);

  // get the geometry and topology from the event setup:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);

  const CaloSubdetectorGeometry *geometry_p;
  CaloSubdetectorTopology *topology_p;

  std::string clustershapetag;
  if (ecalPart == IslandClusterAlgo::barrel) 
    {
      geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
      topology_p = new EcalBarrelTopology(geoHandle);
    }
  else
    {
      geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
      topology_p = new EcalEndcapTopology(geoHandle); 
   }

  const CaloSubdetectorGeometry *geometryES_p;
  geometryES_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

  // Run the clusterization algorithm:
  reco::BasicClusterCollection clusters;
  clusters = island_p->makeClusters(hitCollection_p, geometry_p, topology_p, geometryES_p, ecalPart);

  //Create associated ClusterShape objects.
  std::vector <reco::ClusterShape> ClusVec;
  for (int erg=0;erg<int(clusters.size());++erg){
    reco::ClusterShape TestShape = shapeAlgo_.Calculate(clusters[erg],hitCollection_p,geometry_p,topology_p);
    ClusVec.push_back(TestShape);
  }

  //Put clustershapes in event, but retain a Handle on them.
  std::auto_ptr< reco::ClusterShapeCollection> clustersshapes_p(new reco::ClusterShapeCollection);
  clustersshapes_p->assign(ClusVec.begin(), ClusVec.end());
  edm::OrphanHandle<reco::ClusterShapeCollection> clusHandle; 
  if (ecalPart == IslandClusterAlgo::barrel) 
    clusHandle= evt.put(clustersshapes_p, clustershapecollectionEB_);
  else
    clusHandle= evt.put(clustersshapes_p, clustershapecollectionEE_);

  // create an auto_ptr to a BasicClusterCollection, copy the barrel clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > clusters_p(new reco::BasicClusterCollection);
  clusters_p->assign(clusters.begin(), clusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle;
  if (ecalPart == IslandClusterAlgo::barrel) 
    bccHandle = evt.put(clusters_p, barrelClusterCollection_);
  else
    bccHandle = evt.put(clusters_p, endcapClusterCollection_);


  // BasicClusterShapeAssociationMap
  std::auto_ptr<reco::BasicClusterShapeAssociationCollection> shapeAssocs_p(new reco::BasicClusterShapeAssociationCollection(bccHandle, clusHandle));
  for (unsigned int i = 0; i < clusHandle->size(); i++){
    shapeAssocs_p->insert(edm::Ref<reco::BasicClusterCollection>(bccHandle,i),edm::Ref<reco::ClusterShapeCollection>(clusHandle,i));
  }  
  evt.put(shapeAssocs_p,clusterShapeAssociation);

  delete topology_p;
}
