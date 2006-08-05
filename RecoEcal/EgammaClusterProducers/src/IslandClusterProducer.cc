// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

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
  barrelHitProducer_   = ps.getParameter<std::string>("barrelHitProducer");
  endcapHitProducer_   = ps.getParameter<std::string>("endcapHitProducer");
  barrelHitCollection_ = ps.getParameter<std::string>("barrelHitCollection");
  endcapHitCollection_ = ps.getParameter<std::string>("endcapHitCollection");

  // The names of the produced cluster collections
  barrelClusterCollection_  = ps.getParameter<std::string>("barrelClusterCollection");
  endcapClusterCollection_  = ps.getParameter<std::string>("endcapClusterCollection");

  // Island algorithm parameters
  double barrelSeedThreshold = ps.getParameter<double>("IslandBarrelSeedThr");
  double endcapSeedThreshold = ps.getParameter<double>("IslandEndcapSeedThr");

  clustershape_logweighted = ps.getParameter<bool>("coretools_logweight");
  clustershape_x0 = ps.getParameter<double>("coretools_x0");
  clustershape_t0 = ps.getParameter<double>("coretools_t0");
  clustershape_w0 = ps.getParameter<double>("coretools_w0");

  // Produces a collection of barrel and a collection of endcap clusters
  produces< reco::BasicClusterCollection >(barrelClusterCollection_);
  produces< reco::BasicClusterCollection >(endcapClusterCollection_);

  island_p = new IslandClusterAlgo(barrelSeedThreshold, endcapSeedThreshold, verbosity);

  nEvt_ = 0;
}


IslandClusterProducer::~IslandClusterProducer()
{
  delete island_p;
}


void IslandClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  clusterizeECALPart(evt, es, endcapHitProducer_, endcapHitCollection_, endcapClusterCollection_, IslandClusterAlgo::endcap); 
  clusterizeECALPart(evt, es, barrelHitProducer_, barrelHitCollection_, barrelClusterCollection_, IslandClusterAlgo::barrel);

  nEvt_++;
}


const EcalRecHitCollection * IslandClusterProducer::getCollection(edm::Event& evt,
                                                                  const std::string& hitProducer_,
                                                                  const std::string& hitCollection_)
{
  edm::Handle<EcalRecHitCollection> rhcHandle;
  try
    {
      evt.getByLabel(hitProducer_, hitCollection_, rhcHandle);
      if (!(rhcHandle.isValid())) 
	{
	  std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
	  return 0;
	}
    }
  catch ( cms::Exception& ex ) 
    {
      edm::LogError("IslandClusterProducerError") << "Error! can't get the product " << hitCollection_.c_str() ;
      return 0;
    }
  return rhcHandle.product();
}


void IslandClusterProducer::makeRecHitsMap(std::map<DetId, EcalRecHit> &rechits_m, const EcalRecHitCollection *hitCollection)
{
  EcalRecHitCollection::const_iterator it;
  for (it = hitCollection->begin(); it != hitCollection->end(); it++)
    {
      //Make the map of DetID, EcalRecHit pairs
      rechits_m.insert(std::make_pair(it->id(), *it));    
    }
}


void IslandClusterProducer::clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
                                               const std::string& hitProducer,
                                               const std::string& hitCollection,
                                               const std::string& clusterCollection,
                                               const IslandClusterAlgo::EcalPart& ecalPart)
{
  // get the hit collection from the event:
  const EcalRecHitCollection *hitCollection_p = getCollection(evt, hitProducer, hitCollection);

  // make the map of rechits:
  std::map<DetId, EcalRecHit> rechits_m;
  makeRecHitsMap(rechits_m, hitCollection_p);

  // get the geometry and topology from the event setup:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);

  const CaloSubdetectorGeometry *geometry_p;
  CaloSubdetectorTopology *topology_p;

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

  // Parameters for the position calculation:
  std::map<std::string,double> providedParameters;
  providedParameters.insert(std::make_pair("LogWeighted",clustershape_logweighted));
  providedParameters.insert(std::make_pair("X0",clustershape_x0));
  providedParameters.insert(std::make_pair("T0",clustershape_t0));
  providedParameters.insert(std::make_pair("W0",clustershape_w0));
  PositionCalc::Initialize(providedParameters, &rechits_m, geometry_p);

  // Run the clusterization algorithm:
  reco::BasicClusterCollection clusters;
  clusters = island_p->makeClusters(&rechits_m, geometry_p, topology_p, ecalPart);

  // create an auto_ptr to a BasicClusterCollection, copy the barrel clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > clusters_p(new reco::BasicClusterCollection);
  clusters_p->assign(clusters.begin(), clusters.end());
  evt.put(clusters_p, clusterCollection);

  delete topology_p;
}
