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
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
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
#include "RecoEcal/EgammaClusterProducers/interface/CosmicClusterProducer.h"

CosmicClusterProducer::CosmicClusterProducer(const edm::ParameterSet& ps)
{
  // The verbosity level
  std::string verbosityString = ps.getParameter<std::string>("VerbosityLevel");
  if      (verbosityString == "DEBUG")   verbosity = CosmicClusterAlgo::pDEBUG;
  else if (verbosityString == "WARNING") verbosity = CosmicClusterAlgo::pWARNING;
  else if (verbosityString == "INFO")    verbosity = CosmicClusterAlgo::pINFO;
  else                                   verbosity = CosmicClusterAlgo::pERROR;

  
  // Parameters to identify the hit collections
  ebHitsToken_    = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("barrelHits"));
  eeHitsToken_    = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("endcapHits"));
  
  ebUHitsToken_    = consumes<EcalUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("barrelUncalibHits"));
  eeUHitsToken_    = consumes<EcalUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("endcapUncalibHits"));
  

  // The names of the produced cluster collections
  barrelClusterCollection_  = ps.getParameter<std::string>("barrelClusterCollection");
  endcapClusterCollection_  = ps.getParameter<std::string>("endcapClusterCollection");

  // Island algorithm parameters
  double barrelSeedThreshold        = ps.getParameter<double>("BarrelSeedThr");
  double barrelSingleThreshold      = ps.getParameter<double>("BarrelSingleThr");
  double barrelSecondThreshold      = ps.getParameter<double>("BarrelSecondThr");
  double barrelSupThreshold         = ps.getParameter<double>("BarrelSupThr");
  double endcapSeedThreshold        = ps.getParameter<double>("EndcapSeedThr");
  double endcapSingleThreshold      = ps.getParameter<double>("EndcapSingleThr");
  double endcapSecondThreshold      = ps.getParameter<double>("EndcapSecondThr");
  double endcapSupThreshold         = ps.getParameter<double>("EndcapSupThr");

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

  island_p = new CosmicClusterAlgo(barrelSeedThreshold, barrelSingleThreshold, barrelSecondThreshold, barrelSupThreshold, endcapSeedThreshold, endcapSingleThreshold, endcapSecondThreshold, endcapSupThreshold, posCalculator_,verbosity);

  nEvt_ = 0;
}


CosmicClusterProducer::~CosmicClusterProducer()
{
  delete island_p;
}
 

void CosmicClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  clusterizeECALPart(evt, es, eeHitsToken_,eeUHitsToken_, endcapClusterCollection_, endcapClusterShapeAssociation_, CosmicClusterAlgo::endcap); 
  clusterizeECALPart(evt, es, eeHitsToken_,eeUHitsToken_, barrelClusterCollection_, barrelClusterShapeAssociation_, CosmicClusterAlgo::barrel);
  nEvt_++;
}




void CosmicClusterProducer::clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
                                               const edm::EDGetTokenT<EcalRecHitCollection>& hitsToken,
                                               const edm::EDGetTokenT<EcalUncalibratedRecHitCollection>& uhitsToken,             
                                               const std::string& clusterCollection,
					       const std::string& clusterShapeAssociation,
                                               const CosmicClusterAlgo::EcalPart& ecalPart)
{
  // get the hit collection from the event:

  edm::Handle<EcalRecHitCollection> hits_h;
  edm::Handle<EcalUncalibratedRecHitCollection> uhits_h;
  
  evt.getByToken(hitsToken,hits_h);
  evt.getByToken(uhitsToken,uhits_h);

  const EcalRecHitCollection *hitCollection_p = hits_h.product();
  const EcalUncalibratedRecHitCollection *uhitCollection_p = uhits_h.product();

  // get the geometry and topology from the event setup:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);

  const CaloSubdetectorGeometry *geometry_p;
  CaloSubdetectorTopology *topology_p;

  std::string clustershapetag;
  if (ecalPart == CosmicClusterAlgo::barrel) 
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
  clusters = island_p->makeClusters(hitCollection_p, uhitCollection_p, geometry_p, topology_p, geometryES_p,  ecalPart);
  
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
  if (ecalPart == CosmicClusterAlgo::barrel) 
    clusHandle= evt.put(clustersshapes_p, clustershapecollectionEB_);
  else
    clusHandle= evt.put(clustersshapes_p, clustershapecollectionEE_);
  
  // create an auto_ptr to a BasicClusterCollection, copy the barrel clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > clusters_p(new reco::BasicClusterCollection);
  clusters_p->assign(clusters.begin(), clusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle;
  
  if (ecalPart == CosmicClusterAlgo::barrel) 
    bccHandle = evt.put(clusters_p, barrelClusterCollection_);
  else
    bccHandle = evt.put(clusters_p, endcapClusterCollection_);

  
  // BasicClusterShapeAssociationMap 
  std::auto_ptr<reco::BasicClusterShapeAssociationCollection> shapeAssocs_p(
    new reco::BasicClusterShapeAssociationCollection(bccHandle, clusHandle));

  for (unsigned int i = 0; i < clusHandle->size(); i++){
    shapeAssocs_p->insert(edm::Ref<reco::BasicClusterCollection>(bccHandle,i),edm::Ref<reco::ClusterShapeCollection>(clusHandle,i));
  }  
  evt.put(shapeAssocs_p,clusterShapeAssociation);
  
  delete topology_p;
}
