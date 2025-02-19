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
#include "CommonTools/Utils/interface/StringToEnumValue.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

// Level 1 Trigger
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"

// Class header file
#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTMulti5x5ClusterProducer.h"

EgammaHLTMulti5x5ClusterProducer::EgammaHLTMulti5x5ClusterProducer(const edm::ParameterSet& ps)
{

  doBarrel_   = ps.getParameter<bool>("doBarrel");
  doEndcaps_   = ps.getParameter<bool>("doEndcaps");
  doIsolated_   = ps.getParameter<bool>("doIsolated");

  // Parameters to identify the hit collections
  barrelHitProducer_   = ps.getParameter<edm::InputTag>("barrelHitProducer");
  endcapHitProducer_   = ps.getParameter<edm::InputTag>("endcapHitProducer");
  barrelHitCollection_ = ps.getParameter<std::string>("barrelHitCollection");
  endcapHitCollection_ = ps.getParameter<std::string>("endcapHitCollection");

  // The names of the produced cluster collections
  barrelClusterCollection_  = ps.getParameter<std::string>("barrelClusterCollection");
  endcapClusterCollection_  = ps.getParameter<std::string>("endcapClusterCollection");

  // Multi5x5 algorithm parameters
  double barrelSeedThreshold = ps.getParameter<double>("Multi5x5BarrelSeedThr");
  double endcapSeedThreshold = ps.getParameter<double>("Multi5x5EndcapSeedThr");

  // L1 matching parameters
  l1TagIsolated_ = ps.getParameter< edm::InputTag > ("l1TagIsolated");
  l1TagNonIsolated_ = ps.getParameter< edm::InputTag > ("l1TagNonIsolated");
  l1LowerThr_ = ps.getParameter<double> ("l1LowerThr");
  l1UpperThr_ = ps.getParameter<double> ("l1UpperThr");
  l1LowerThrIgnoreIsolation_ = ps.getParameter<double> ("l1LowerThrIgnoreIsolation");

  regionEtaMargin_   = ps.getParameter<double>("regionEtaMargin");
  regionPhiMargin_   = ps.getParameter<double>("regionPhiMargin");

  // Parameters for the position calculation:
  posCalculator_ = PositionCalc( ps.getParameter<edm::ParameterSet>("posCalcParameters") );

  const std::vector<std::string> flagnames = 
    ps.getParameter<std::vector<std::string> >("RecHitFlagToBeExcluded");

  // exclude recHit flags from seeding
  std::vector<int> v_chstatus = StringToEnumValue<EcalRecHit::Flags>(flagnames);
  

  // Produces a collection of barrel and a collection of endcap clusters

  produces< reco::BasicClusterCollection >(endcapClusterCollection_);
  produces< reco::BasicClusterCollection >(barrelClusterCollection_);

  Multi5x5_p = new Multi5x5ClusterAlgo(barrelSeedThreshold, endcapSeedThreshold, v_chstatus, posCalculator_);
  
  /*
  shapeAlgo_ = ClusterShapeAlgo(providedParameters);//new


  clustershapecollectionEB_ = ps.getParameter<std::string>("clustershapecollectionEB");
  clustershapecollectionEE_ = ps.getParameter<std::string>("clustershapecollectionEE");

  //AssociationMap
  barrelClusterShapeAssociation_ = ps.getParameter<std::string>("barrelShapeAssociation");
  endcapClusterShapeAssociation_ = ps.getParameter<std::string>("endcapShapeAssociation");
  
  // Produces a collection of barrel and a collection of endcap clusters

  produces< reco::ClusterShapeCollection>(clustershapecollectionEE_);//new
  //produces< reco::BasicClusterCollection >(endcapClusterCollection_);
  produces< reco::ClusterShapeCollection>(clustershapecollectionEB_);//new
 // produces< reco::BasicClusterCollection >(barrelClusterCollection_);
  produces< reco::BasicClusterShapeAssociationCollection >(barrelClusterShapeAssociation_);//new
  produces< reco::BasicClusterShapeAssociationCollection >(endcapClusterShapeAssociation_);//new

  */

  nEvt_ = 0;
}


EgammaHLTMulti5x5ClusterProducer::~EgammaHLTMulti5x5ClusterProducer()
{
  delete Multi5x5_p;
}


void EgammaHLTMulti5x5ClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emIsolColl ;
  if(doIsolated_)
    evt.getByLabel(l1TagIsolated_, emIsolColl);
  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emNonIsolColl ;
  evt.getByLabel(l1TagNonIsolated_, emNonIsolColl);
  // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  es.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;

  std::vector<EcalEtaPhiRegion> barrelRegions;
  std::vector<EcalEtaPhiRegion> endcapRegions;

  if(doIsolated_) {
    for( l1extra::L1EmParticleCollection::const_iterator emItr = emIsolColl->begin(); emItr != emIsolColl->end() ;++emItr ){

      if (emItr->et() > l1LowerThr_ && emItr->et() < l1UpperThr_) {
	
	// Access the GCT hardware object corresponding to the L1Extra EM object.
	int etaIndex = emItr->gctEmCand()->etaIndex() ;
	
	
	int phiIndex = emItr->gctEmCand()->phiIndex() ;
	// Use the L1CaloGeometry to find the eta, phi bin boundaries.
	double etaLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
	double etaHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
	double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
	double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;

	//Attention isForward does not work
	int isforw=0;
	int isbarl=0;
	if((float)(etaHigh)>1.479 || (float)(etaLow)<-1.479) isforw=1;
	if(((float)(etaLow)>-1.479 && (float)(etaLow)<1.479) || 
	   ((float)(etaHigh)>-1.479 && (float)(etaHigh)<1.479)) isbarl=1;

	//std::cout<<"Multi5x5 etaindex "<<etaIndex<<" low hig : "<<etaLow<<" "<<etaHigh<<" phi low hig" <<phiLow<<" " << phiHigh<<" isforw "<<emItr->gctEmCand()->regionId().isForward()<<" isforwnew" <<isforw<< std::endl;
	
	etaLow -= regionEtaMargin_;
	etaHigh += regionEtaMargin_;
	phiLow -= regionPhiMargin_;
	phiHigh += regionPhiMargin_;

	//if (emItr->gctEmCand()->regionId().isForward()) {
	if (isforw) {
	  if (etaHigh>-1.479 && etaHigh<1.479) etaHigh=-1.479;
	  if ( etaLow>-1.479 &&  etaLow<1.479) etaLow=1.479;
	  EcalEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  endcapRegions.push_back(region);
	}
	if (isbarl) {
	  if (etaHigh>1.479) etaHigh=1.479;
	  if (etaLow<-1.479) etaLow=-1.479;
	  EcalEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  barrelRegions.push_back(region);
	}
	EcalEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	
      }
    }
  }


  if(!doIsolated_||l1LowerThrIgnoreIsolation_<64) {
    for( l1extra::L1EmParticleCollection::const_iterator emItr = emNonIsolColl->begin(); emItr != emNonIsolColl->end() ;++emItr ){

      if(doIsolated_&&emItr->et()<l1LowerThrIgnoreIsolation_) continue;

      if (emItr->et() > l1LowerThr_ && emItr->et() < l1UpperThr_) {
	
	// Access the GCT hardware object corresponding to the L1Extra EM object.
	int etaIndex = emItr->gctEmCand()->etaIndex() ;
	
	
	int phiIndex = emItr->gctEmCand()->phiIndex() ;
	// Use the L1CaloGeometry to find the eta, phi bin boundaries.
	double etaLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
	double etaHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
	double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
	double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;


	int isforw=0;
	int isbarl=0;
	if((float)(etaHigh)>1.479 || (float)(etaLow)<-1.479) isforw=1;
	if(((float)(etaLow)>-1.479 && (float)(etaLow)<1.479) || 
	   ((float)(etaHigh)>-1.479 && (float)(etaHigh)<1.479)) isbarl=1;

	//std::cout<<"Multi5x5 etaindex "<<etaIndex<<" low hig : "<<etaLow<<" "<<etaHigh<<" phi low hig" <<phiLow<<" " << phiHigh<<" isforw "<<emItr->gctEmCand()->regionId().isForward()<<" isforwnew" <<isforw<< std::endl;
	
	etaLow -= regionEtaMargin_;
	etaHigh += regionEtaMargin_;
	phiLow -= regionPhiMargin_;
	phiHigh += regionPhiMargin_;

	//if (emItr->gctEmCand()->regionId().isForward()) {
	if (isforw) {
	  if (etaHigh>-1.479 && etaHigh<1.479) etaHigh=-1.479;
	  if ( etaLow>-1.479 &&  etaLow<1.479) etaLow=1.479;
	  EcalEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  endcapRegions.push_back(region);
	}
	if (isbarl) {
	  if (etaHigh>1.479) etaHigh=1.479;
	  if (etaLow<-1.479) etaLow=-1.479;
	  EcalEtaPhiRegion region(etaLow,etaHigh,phiLow,phiHigh);
	  barrelRegions.push_back(region);
	}
	
      }
    }
  }


  if (doEndcaps_ 
      //&&endcapRegions.size()!=0
      ) {

    clusterizeECALPart(evt, es, endcapHitProducer_.label(), endcapHitCollection_, endcapClusterCollection_, endcapRegions, reco::CaloID::DET_ECAL_ENDCAP);//old
   }
  if (doBarrel_ 
      //&& barrelRegions.size()!=0
      ) {
    clusterizeECALPart(evt, es, barrelHitProducer_.label(), barrelHitCollection_, barrelClusterCollection_, barrelRegions, reco::CaloID::DET_ECAL_BARREL);//old
 
  }
  nEvt_++;
}


const EcalRecHitCollection * EgammaHLTMulti5x5ClusterProducer::getCollection(edm::Event& evt,
                                                                  const std::string& hitProducer_,
                                                                  const std::string& hitCollection_)
{
  edm::Handle<EcalRecHitCollection> rhcHandle;

  evt.getByLabel(hitProducer_, hitCollection_, rhcHandle);
  if (!(rhcHandle.isValid())) 
    {
      std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
      edm::LogError("EgammaHLTMulti5x5ClusterProducerError") << "Error! can't get the product " << hitCollection_.c_str() ;
      return 0;
    } 
  return rhcHandle.product();
}


void EgammaHLTMulti5x5ClusterProducer::clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
                                               const std::string& hitProducer,
                                               const std::string& hitCollection,
                                               const std::string& clusterCollection,
                                               const std::vector<EcalEtaPhiRegion>& regions,
                                                const reco::CaloID::Detectors detector)
{

  // get the hit collection from the event:
  const EcalRecHitCollection *hitCollection_p = getCollection(evt, hitProducer, hitCollection);

  // get the geometry and topology from the event setup:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);

  const CaloSubdetectorGeometry *geometry_p;
  CaloSubdetectorTopology *topology_p;

  if (detector == reco::CaloID::DET_ECAL_BARREL) 
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
  clusters = Multi5x5_p->makeClusters(hitCollection_p, geometry_p, topology_p, geometryES_p, detector, true, regions);

  // create an auto_ptr to a BasicClusterCollection, copy the barrel clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > clusters_p(new reco::BasicClusterCollection);
  clusters_p->assign(clusters.begin(), clusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle;
  if (detector == reco::CaloID::DET_ECAL_BARREL) 
    bccHandle = evt.put(clusters_p, barrelClusterCollection_);
  else
    bccHandle = evt.put(clusters_p, endcapClusterCollection_);

  delete topology_p;
}
