// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/HybridClusterProducer.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

 

HybridClusterProducer::HybridClusterProducer(const edm::ParameterSet& ps)
{


  basicclusterCollection_ = ps.getParameter<std::string>("basicclusterCollection");
  superclusterCollection_ = ps.getParameter<std::string>("superclusterCollection");
  hitproducer_ = ps.getParameter<std::string>("ecalhitproducer");
  hitcollection_ =ps.getParameter<std::string>("ecalhitcollection");
   
  //Setup for core tools objects. 
  edm::ParameterSet posCalcParameters = 
    ps.getParameter<edm::ParameterSet>("posCalcParameters");

  posCalculator_ = PositionCalc(posCalcParameters);

  const std::vector<std::string> flagnames = 
    ps.getParameter<std::vector<std::string> >("RecHitFlagToBeExcluded");

  const std::vector<int> flagsexcl= 
    StringToEnumValue<EcalRecHit::Flags>(flagnames);

  const std::vector<std::string> severitynames = 
    ps.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcluded");

  const std::vector<int> severitiesexcl= 
    StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynames);

  hybrid_p = new HybridClusterAlgo(ps.getParameter<double>("HybridBarrelSeedThr"), 
                                   ps.getParameter<int>("step"),
                                   ps.getParameter<double>("ethresh"),
                                   ps.getParameter<double>("eseed"),
                                   ps.getParameter<double>("xi"),
                                   ps.getParameter<bool>("useEtForXi"),
                                   ps.getParameter<double>("ewing"),
				   flagsexcl,
                                   posCalculator_,
			           ps.getParameter<bool>("dynamicEThresh"),
                                   ps.getParameter<double>("eThreshA"),
                                   ps.getParameter<double>("eThreshB"),
				   severitiesexcl,
				   ps.getParameter<bool>("excludeFlagged")
                                   );
                                   //bremRecoveryPset,

  // get brem recovery parameters
  bool dynamicPhiRoad = ps.getParameter<bool>("dynamicPhiRoad");
  if (dynamicPhiRoad) {
     edm::ParameterSet bremRecoveryPset = ps.getParameter<edm::ParameterSet>("bremRecoveryPset");
     hybrid_p->setDynamicPhiRoad(bremRecoveryPset);
  }

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
  if (!(rhcHandle.isValid())){
    edm::LogError("MissingProduct") << "could not get a handle on the EcalRecHitCollection!";
    return;

  }
  const EcalRecHitCollection *hit_collection = rhcHandle.product();

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry *geometry_p;
  std::auto_ptr<const CaloSubdetectorTopology> topology;

  edm::ESHandle<EcalSeverityLevelAlgo> sevLv;
  es.get<EcalSeverityLevelAlgoRcd>().get(sevLv);

  LogTrace("EcalClusters") << "\n\n\n" << hitcollection_ << "\n\n";

  if(hitcollection_ == "EcalRecHitsEB") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    topology.reset(new EcalBarrelTopology(geoHandle));
  } else if(hitcollection_ == "EcalRecHitsEE") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    topology.reset(new EcalEndcapTopology(geoHandle));
  } else if(hitcollection_ == "EcalRecHitsPS") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
    topology.reset(new EcalPreshowerTopology (geoHandle));
  } else throw(std::runtime_error("\n\nHybrid Cluster Producer encountered invalied ecalhitcollection type.\n\n"));
    
  // make the Basic clusters!
  reco::BasicClusterCollection basicClusters;
  hybrid_p->makeClusters(hit_collection, geometry_p, basicClusters, sevLv.product(),false,
			 std::vector<EcalEtaPhiRegion>());

  LogTrace("EcalClusters") << "Finished clustering - BasicClusterCollection returned to producer..." ;

  // create an auto_ptr to a BasicClusterCollection, copy the clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > basicclusters_p(new reco::BasicClusterCollection);
  basicclusters_p->assign(basicClusters.begin(), basicClusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle =  evt.put(basicclusters_p,basicclusterCollection_);
								       
  //Basic clusters now in the event.
  LogTrace("EcalClusters") << "Basic Clusters now put into event." ;
  
  
  //Weird though it is, get the BasicClusters back out of the event.  We need the
  //edm::Ref to these guys to make our superclusters for Hybrid.
  //edm::Handle<reco::BasicClusterCollection> bccHandle;
  // evt.getByLabel("clusterproducer",basicclusterCollection_, bccHandle);
  if (!(bccHandle.isValid())) {
    edm::LogError("Missing Product") << "could not get a handle on the BasicClusterCollection!" ;
    return;
  }
 
  reco::BasicClusterCollection clusterCollection = *bccHandle;
  
  LogTrace("EcalClusters")<< "Got the BasicClusterCollection" << std::endl;

  reco::CaloClusterPtrVector clusterPtrVector;
  for (unsigned int i = 0; i < clusterCollection.size(); i++){
    clusterPtrVector.push_back(reco::CaloClusterPtr(bccHandle, i));
  }

  reco::SuperClusterCollection superClusters = hybrid_p->makeSuperClusters(clusterPtrVector);
  LogTrace("EcalClusters") << "Found: " << superClusters.size() << " superclusters." ;

  std::auto_ptr< reco::SuperClusterCollection > superclusters_p(new reco::SuperClusterCollection);
  superclusters_p->assign(superClusters.begin(), superClusters.end());
  
  evt.put(superclusters_p, superclusterCollection_);
  LogTrace("EcalClusters") << "Hybrid Clusters (Basic/Super) added to the Event! :-)" ;

  
  nEvt_++;
}

