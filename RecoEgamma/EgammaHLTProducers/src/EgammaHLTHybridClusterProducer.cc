// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"

// Level 1 Trigger
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"

// Class header file
#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTHybridClusterProducer.h"


EgammaHLTHybridClusterProducer::EgammaHLTHybridClusterProducer(const edm::ParameterSet& ps) {

  basicclusterCollection_ = ps.getParameter<std::string>("basicclusterCollection");
  superclusterCollection_ = ps.getParameter<std::string>("superclusterCollection");
  hitcollection_ = ps.getParameter<edm::InputTag>("ecalhitcollection");
  hittoken_ = consumes<EcalRecHitCollection>(hitcollection_);
  
  // L1 matching parameters
  l1TagIsolated_    = consumes<l1extra::L1EmParticleCollection>(ps.getParameter< edm::InputTag > ("l1TagIsolated"));
  l1TagNonIsolated_ = consumes<l1extra::L1EmParticleCollection>(ps.getParameter< edm::InputTag > ("l1TagNonIsolated"));

  doIsolated_   = ps.getParameter<bool>("doIsolated");

  l1LowerThr_ = ps.getParameter<double> ("l1LowerThr");
  l1UpperThr_ = ps.getParameter<double> ("l1UpperThr");
  l1LowerThrIgnoreIsolation_ = ps.getParameter<double> ("l1LowerThrIgnoreIsolation");

  regionEtaMargin_   = ps.getParameter<double>("regionEtaMargin");
  regionPhiMargin_   = ps.getParameter<double>("regionPhiMargin");

  // Parameters for the position calculation:
  posCalculator_ = PositionCalc( ps.getParameter<edm::ParameterSet>("posCalcParameters") );
  
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

  bool dynamicPhiRoad = ps.getParameter<bool>("dynamicPhiRoad");
    if (dynamicPhiRoad) {
     edm::ParameterSet bremRecoveryPset = ps.getParameter<edm::ParameterSet>("bremRecoveryPset");
     hybrid_p->setDynamicPhiRoad(bremRecoveryPset);
  }

  produces< reco::BasicClusterCollection >(basicclusterCollection_);
  produces< reco::SuperClusterCollection >(superclusterCollection_);
  nEvt_ = 0;
}


EgammaHLTHybridClusterProducer::~EgammaHLTHybridClusterProducer()
{
  delete hybrid_p;
}

void EgammaHLTHybridClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<std::string>("debugLevel" , "INFO");
  desc.add<std::string>("basicclusterCollection", "");
  desc.add<std::string>("superclusterCollection", "");
  desc.add<edm::InputTag>("ecalhitcollection", edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  desc.add<edm::InputTag>("l1TagIsolated", edm::InputTag("l1extraParticles","Isolated"));
  desc.add<edm::InputTag>("l1TagNonIsolated", edm::InputTag("l1extraParticles","NonIsolated"));
  desc.add<bool>("doIsolated", true);
  desc.add<double>("l1LowerThr", 0);
  desc.add<double>("l1UpperThr", 9999.0);
  desc.add<double>("l1LowerThrIgnoreIsolation", 999.0);
  desc.add<double>("regionEtaMargin", 0.14);
  desc.add<double>("regionPhiMargin", 0.4);

  edm::ParameterSetDescription posCalcPSET;
  posCalcPSET.add<double>("T0_barl", 7.4);
  posCalcPSET.add<double>("T0_endc", 3.1);
  posCalcPSET.add<double>("T0_endcPresh", 1.2);
  posCalcPSET.add<double>("W0", 4.2);
  posCalcPSET.add<double>("X0", 0.89);
  posCalcPSET.add<bool>("LogWeighted", true);
  desc.add<edm::ParameterSetDescription>("posCalcParameters", posCalcPSET);

  desc.add<std::vector<std::string>>("RecHitFlagToBeExcluded", std::vector<std::string>());
  desc.add<std::vector<std::string> >("RecHitSeverityToBeExcluded", std::vector<std::string>());
  desc.add<double>("severityRecHitThreshold", 4.0);
  desc.add<double>("HybridBarrelSeedThr", 1.0);
  desc.add<int>("step", 10);
  desc.add<double>("ethresh", 0.1);
  desc.add<double>("eseed", 0.35);
  desc.add<double>("xi", 0);
  desc.add<bool>("useEtForXi", true);
  desc.add<double>("ewing", 1.0);
  desc.add<bool>("dynamicEThresh", false);
  desc.add<double>("eThreshA", 0.003);
  desc.add<double>("eThreshB", 0.1);
  desc.add<bool>("excludeFlagged", false);
  desc.add<bool>("dynamicPhiRoad", false);
  //desc.add<edm::ParameterSet>("bremRecoveryPset", edm::ParameterSet());
  
  descriptions.add("hltEgammaHLTHybridClusterProducer", desc);  
}


void EgammaHLTHybridClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByToken(hittoken_, rhcHandle);
  
  if (!(rhcHandle.isValid()))  
    {
      edm::LogError("ProductNotFound")<< "could not get a handle on the EcalRecHitCollection!" << std::endl;
      return;
    }
  const EcalRecHitCollection *hit_collection = rhcHandle.product();

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry *geometry_p;
  std::auto_ptr<const CaloSubdetectorTopology> topology;

  //edm::ESHandle<EcalChannelStatus> chStatus;
  //es.get<EcalChannelStatusRcd>().get(chStatus);
  //const EcalChannelStatus* theEcalChStatus = chStatus.product();
  
  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  es.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  const EcalSeverityLevelAlgo* sevLevel = sevlv.product();
 
  if(hitcollection_.instance() == "EcalRecHitsEB") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    topology.reset(new EcalBarrelTopology(geoHandle));
  } else if(hitcollection_.instance() == "EcalRecHitsEE") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    topology.reset(new EcalEndcapTopology(geoHandle));
  } else if(hitcollection_.instance() == "EcalRecHitsPS") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
    topology.reset(new EcalPreshowerTopology (geoHandle));
  } else throw(std::runtime_error("\n\nHybrid Cluster Producer encountered invalied ecalhitcollection type.\n\n"));
    
  //Get the L1 EM Particle Collection
  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emIsolColl ;
  if(doIsolated_)
    evt.getByToken(l1TagIsolated_, emIsolColl);

  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emNonIsolColl ;
  evt.getByToken(l1TagNonIsolated_, emNonIsolColl);

  // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  es.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;

  std::vector<EcalEtaPhiRegion> regions;

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

       int isbarl=0;
       //Part of the region is in the barel if either the upper or lower
       //edge of the region is within the barrel
       if(((float)(etaLow)>-1.479 && (float)(etaLow)<1.479) || 
	  ((float)(etaHigh)>-1.479 && (float)(etaHigh)<1.479)) isbarl=1;


      etaLow -= regionEtaMargin_;
      etaHigh += regionEtaMargin_;
      phiLow -= regionPhiMargin_;
      phiHigh += regionPhiMargin_;

      if (etaHigh>1.479) etaHigh=1.479;
      if (etaLow<-1.479) etaLow=-1.479;

      if(isbarl) regions.push_back(EcalEtaPhiRegion(etaLow,etaHigh,phiLow,phiHigh));

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

       int isbarl=0;
       //Part of the region is in the barel if either the upper or lower
       //edge of the region is within the barrel
       if(((float)(etaLow)>-1.479 && (float)(etaLow)<1.479) || 
          ((float)(etaHigh)>-1.479 && (float)(etaHigh)<1.479)) isbarl=1;
       
       
       etaLow -= regionEtaMargin_;
       etaHigh += regionEtaMargin_;
       phiLow -= regionPhiMargin_;
       phiHigh += regionPhiMargin_;
       
       if (etaHigh>1.479) etaHigh=1.479;
       if (etaLow<-1.479) etaLow=-1.479;
       
       if(isbarl) regions.push_back(EcalEtaPhiRegion(etaLow,etaHigh,phiLow,phiHigh));
       
    }
    }
  }
  
  // make the Basic clusters!
  reco::BasicClusterCollection basicClusters;
  hybrid_p->makeClusters(hit_collection, geometry_p, basicClusters, sevLevel, true, regions);
  
  // create an auto_ptr to a BasicClusterCollection, copy the clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > basicclusters_p(new reco::BasicClusterCollection);
  basicclusters_p->assign(basicClusters.begin(), basicClusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle =  evt.put(basicclusters_p, 
                                                                       basicclusterCollection_);
  if (!(bccHandle.isValid())) {
    return;
  }
  reco::BasicClusterCollection clusterCollection = *bccHandle;

  reco::CaloClusterPtrVector clusterRefVector;
  for (unsigned int i = 0; i < clusterCollection.size(); i++){
    clusterRefVector.push_back(reco::CaloClusterPtr(bccHandle, i));
  }

  reco::SuperClusterCollection superClusters = hybrid_p->makeSuperClusters(clusterRefVector);

  std::auto_ptr< reco::SuperClusterCollection > superclusters_p(new reco::SuperClusterCollection);
  superclusters_p->assign(superClusters.begin(), superClusters.end());
  evt.put(superclusters_p, superclusterCollection_);


  nEvt_++;
}

 
