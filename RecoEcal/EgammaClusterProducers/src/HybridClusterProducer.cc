// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/HybridClusterProducer.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

 

HybridClusterProducer::HybridClusterProducer(const edm::ParameterSet& ps)
{

    // The debug level
  std::string debugString = ps.getParameter<std::string>("debugLevel");
  if      (debugString == "DEBUG")   debugL = HybridClusterAlgo::pDEBUG;
  else if (debugString == "INFO")    debugL = HybridClusterAlgo::pINFO;
  else                               debugL = HybridClusterAlgo::pERROR;
  hybrid_p = new HybridClusterAlgo(ps.getParameter<double>("HybridBarrelSeedThr"), 
				   ps.getParameter<int>("step"),
				   ps.getParameter<double>("ethresh"),
				   ps.getParameter<double>("eseed"),
				   ps.getParameter<double>("ewing"),
				   debugL);

  basicclusterCollection_ = ps.getParameter<std::string>("basicclusterCollection");
  superclusterCollection_ = ps.getParameter<std::string>("superclusterCollection");
  hitproducer_ = ps.getParameter<std::string>("ecalhitproducer");
  hitcollection_ =ps.getParameter<std::string>("ecalhitcollection");
  clustershape_logweighted = ps.getParameter<bool>("coretools_logweight");
  clustershape_x0 = ps.getParameter<double>("coretools_x0");
  clustershape_t0 = ps.getParameter<double>("coretools_t0");
  clustershape_w0 = ps.getParameter<double>("coretools_w0");
  clustershapecollection_ = ps.getParameter<std::string>("clustershapecollection");

  produces< reco::ClusterShapeCollection>(clustershapecollection_);
  produces< reco::BasicClusterCollection >(basicclusterCollection_);
  produces< reco::SuperClusterCollection >(superclusterCollection_);
  nEvt_ = 0;
}


HybridClusterProducer::~HybridClusterProducer()
{
  delete hybrid_p;
}


void HybridClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;
  //  evt.getByType(rhcHandle);
  evt.getByLabel(hitproducer_, hitcollection_, rhcHandle);
  if (!(rhcHandle.isValid())) 
    {
      if (debugL <= HybridClusterAlgo::pINFO)
	std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
      return;
    }
  const EcalRecHitCollection *hit_collection = rhcHandle.product();

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry *geometry_p;

  if (debugL == HybridClusterAlgo::pDEBUG)
    std::cout << "\n\n\n" << hitcollection_ << "\n\n" << std::endl;

  if(hitcollection_ == "EcalRecHitsEB") geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  else if(hitcollection_ == "EcalRecHitsEE") geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  else if(hitcollection_ == "EcalRecHitsPS") geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  else throw(std::runtime_error("\n\nHybrid Cluster Producer encountered invalied ecalhitcollection type.\n\n"));
    
  //Setup for core tools objects.
  std::map<std::string,double> providedParameters;  
  providedParameters.insert(std::make_pair("LogWeighted",clustershape_logweighted));
  providedParameters.insert(std::make_pair("X0",clustershape_x0));
  providedParameters.insert(std::make_pair("T0",clustershape_t0));
  providedParameters.insert(std::make_pair("W0",clustershape_w0));
  PositionCalc::Initialize(providedParameters, hit_collection, &(*geometry_p));
  ClusterShapeAlgo::Initialize(hit_collection, &geoHandle);
  //Done with setup
  
  // make the Basic clusters!
  reco::BasicClusterCollection basicClusters;
  hybrid_p->makeClusters(hit_collection, geometry_p, basicClusters);
  if (debugL == HybridClusterAlgo::pDEBUG)
    std::cout << "Finished clustering - BasicClusterCollection returned to producer..." << std::endl;

  std::vector <reco::ClusterShape> ClusVec;
  for (int erg=0;erg<int(basicClusters.size());++erg){
    reco::ClusterShape TestShape = ClusterShapeAlgo::Calculate(basicClusters[erg]);
    ClusVec.push_back(TestShape);
  }
  std::auto_ptr< reco::ClusterShapeCollection> clustersshapes_p(new reco::ClusterShapeCollection);
  clustersshapes_p->assign(ClusVec.begin(), ClusVec.end());
  edm::OrphanHandle<reco::ClusterShapeCollection> clusHandle = evt.put(clustersshapes_p, 
								       clustershapecollection_);

  reco::ClusterShapeCollection clusColl= *clusHandle;
  for (unsigned int i = 0; i < clusColl.size(); i++){
    reco::ClusterShapeRef reffer(reco::ClusterShapeRef(clusHandle, i));
    basicClusters[i].SetClusterShapeRef(reffer);
  }
  // create an auto_ptr to a BasicClusterCollection, copy the clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > basicclusters_p(new reco::BasicClusterCollection);
  basicclusters_p->assign(basicClusters.begin(), basicClusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle =  evt.put(basicclusters_p, 
                                                                       basicclusterCollection_);
  //Basic clusters now in the event.
  if (debugL == HybridClusterAlgo::pDEBUG)
    std::cout << "Basic Clusters now put into event." << std::endl;
  
  //Weird though it is, get the BasicClusters back out of the event.  We need the
  //edm::Ref to these guys to make our superclusters for Hybrid.
//  edm::Handle<reco::BasicClusterCollection> bccHandle;
 // evt.getByLabel("clusterproducer",basicclusterCollection_, bccHandle);
  if (!(bccHandle.isValid())) {
    if (debugL <= HybridClusterAlgo::pINFO)
      std::cout << "could not get a handle on the BasicClusterCollection!" << std::endl;
    return;
  }
  reco::BasicClusterCollection clusterCollection = *bccHandle;
  if (debugL == HybridClusterAlgo::pDEBUG)
    std::cout << "Got the BasicClusterCollection" << std::endl;

  reco::BasicClusterRefVector clusterRefVector;
  for (unsigned int i = 0; i < clusterCollection.size(); i++){
    clusterRefVector.push_back(reco::BasicClusterRef(bccHandle, i));
  }

  reco::SuperClusterCollection superClusters = hybrid_p->makeSuperClusters(clusterRefVector);
  if (debugL == HybridClusterAlgo::pDEBUG)
    std::cout << "Found: " << superClusters.size() << " superclusters." << std::endl;  
  
  std::auto_ptr< reco::SuperClusterCollection > superclusters_p(new reco::SuperClusterCollection);
  superclusters_p->assign(superClusters.begin(), superClusters.end());
  evt.put(superclusters_p, superclusterCollection_);

  if (debugL == HybridClusterAlgo::pDEBUG)
    std::cout << "Hybrid Clusters (Basic/Super) added to the Event! :-)" << std::endl;

  nEvt_++;
}

