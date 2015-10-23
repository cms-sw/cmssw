// C/C++ headers
//#include <iostream>
//#include <vector>

// Reconstruction Classes
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTNxNClusterProducer.h"
#include "TVector3.h"

#include <memory>

EgammaHLTNxNClusterProducer::EgammaHLTNxNClusterProducer(const edm::ParameterSet& ps):
  doBarrel_               (ps.getParameter<bool>("doBarrel")),
  doEndcaps_              (ps.getParameter<bool>("doEndcaps")),
  barrelHitProducer_      (consumes<EcalRecHitCollection>(ps.getParameter< edm::InputTag > ("barrelHitProducer"))),
  endcapHitProducer_      (consumes<EcalRecHitCollection>(ps.getParameter< edm::InputTag > ("endcapHitProducer"))),
  clusEtaSize_            (ps.getParameter<int> ("clusEtaSize")),
  clusPhiSize_            (ps.getParameter<int> ("clusPhiSize")),
  barrelClusterCollection_(ps.getParameter<std::string>("barrelClusterCollection")),
  endcapClusterCollection_(ps.getParameter<std::string>("endcapClusterCollection")),
  clusSeedThr_            (ps.getParameter<double> ("clusSeedThr")),
  clusSeedThrEndCap_      (ps.getParameter<double> ("clusSeedThrEndCap")),
  useRecoFlag_            (ps.getParameter<bool>("useRecoFlag")),
  flagLevelRecHitsToUse_  (ps.getParameter<int>("flagLevelRecHitsToUse")), 
  useDBStatus_            (ps.getParameter<bool>("useDBStatus")),
  statusLevelRecHitsToUse_(ps.getParameter<int>("statusLevelRecHitsToUse")), 
  maxNumberofSeeds_       (ps.getParameter<int> ("maxNumberofSeeds")),
  maxNumberofClusters_    (ps.getParameter<int> ("maxNumberofClusters")),
  debug_                  (ps.getParameter<int> ("debugLevel")),
  posCalculator_          (PositionCalc( ps.getParameter<edm::ParameterSet>("posCalcParameters"))) {
  
  produces< reco::BasicClusterCollection >(barrelClusterCollection_);
  produces< reco::BasicClusterCollection >(endcapClusterCollection_);
}


EgammaHLTNxNClusterProducer::~EgammaHLTNxNClusterProducer()
{}

void EgammaHLTNxNClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<bool>(("doBarrel"), true);
  desc.add<bool>(("doEndcaps"), true);
  desc.add<edm::InputTag>(("barrelHitProducer"), edm::InputTag("hltEcalRegionalPi0EtaRecHit","EcalRecHitsEB"));
  desc.add<edm::InputTag>(("endcapHitProducer"), edm::InputTag("hltEcalRegionalPi0EtaRecHit","EcalRecHitsEE"));
  desc.add<int>(("clusEtaSize"), 3);
  desc.add<int>(("clusPhiSize"), 3);
  desc.add<std::string>(("barrelClusterCollection"), "Simple3x3ClustersBarrel");
  desc.add<std::string>(("endcapClusterCollection"), "Simple3x3ClustersEndcap");
  desc.add<double>(("clusSeedThr"), 0.5);
  desc.add<double>(("clusSeedThrEndCap"), 1.0);
  desc.add<bool>(("useRecoFlag"), false);
  desc.add<int>(("flagLevelRecHitsToUse"), 1); 
  desc.add<bool>(("useDBStatus"), true);
  desc.add<int>(("statusLevelRecHitsToUse"), 1);

  edm::ParameterSetDescription posCalcPSET;
  posCalcPSET.add<double>("T0_barl", 7.4);
  posCalcPSET.add<double>("T0_endc", 3.1);
  posCalcPSET.add<double>("T0_endcPresh", 1.2);
  posCalcPSET.add<double>("W0", 4.2);
  posCalcPSET.add<double>("X0", 0.89);
  posCalcPSET.add<bool>("LogWeighted", true);
  desc.add<edm::ParameterSetDescription>("posCalcParameters", posCalcPSET);
 
  desc.add<int>(("maxNumberofSeeds"), 1000);
  desc.add<int>(("maxNumberofClusters"), 200);
  desc.add<int>(("debugLevel"), 0);
  descriptions.add(("hltEgammaHLTNxNClusterProducer"), desc);  
}

void EgammaHLTNxNClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
    
  if(doBarrel_) {
    edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
    evt.getByToken(barrelHitProducer_, barrelRecHitsHandle);
    if (!barrelRecHitsHandle.isValid()) {
      LogDebug("") << "EgammaHLTNxNClusterProducer Error! can't get product eb hit!" << std::endl;
    }
    
    const EcalRecHitCollection *hits_eb = barrelRecHitsHandle.product();
    if( debug_>=2 )
      LogDebug("")<<"EgammaHLTNxNClusterProducer nEBrechits: "<< evt.id().run()<<" event "<<evt.id().event() <<" "<< hits_eb->size()<<std::endl;
    
    makeNxNClusters(evt,es,hits_eb, reco::CaloID::DET_ECAL_BARREL);
    
  }
  
  
  if(doEndcaps_) {
    edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
    evt.getByToken(endcapHitProducer_, endcapRecHitsHandle);
    if (!endcapRecHitsHandle.isValid()) {
      LogDebug("") << "EgammaHLTNxNClusterProducer Error! can't get product ee hit!" << std::endl;
    }
    
    const EcalRecHitCollection *hits_ee = endcapRecHitsHandle.product();
    if( debug_>=2 ) 
      LogDebug("")<<"EgammaHLTNxNClusterProducer nEErechits: "<< evt.id().run()<<" event "<<evt.id().event() <<" "<< hits_ee->size()<<std::endl;
    makeNxNClusters(evt,es,hits_ee, reco::CaloID::DET_ECAL_ENDCAP);
  }
}



bool EgammaHLTNxNClusterProducer::checkStatusOfEcalRecHit(const EcalChannelStatus &channelStatus,const EcalRecHit &rh){

  if(useRecoFlag_ ){ ///from recoFlag()
    int flag = rh.recoFlag();
    if( flagLevelRecHitsToUse_ ==0){ ///good 
      if( flag != 0) return false; 
    }
    else if( flagLevelRecHitsToUse_ ==1){ ///good || PoorCalib 
      if( flag !=0 && flag != 4 ) return false; 
    }
    else if( flagLevelRecHitsToUse_ ==2){ ///good || PoorCalib || LeadingEdgeRecovered || kNeighboursRecovered,
      if( flag !=0 && flag != 4 && flag != 6 && flag != 7) return false; 
    }
  }
  if ( useDBStatus_){ //// from DB
    int status =  int(channelStatus[rh.id().rawId()].getStatusCode()); 
    if ( status > statusLevelRecHitsToUse_ ) return false; 
  }
  
  return true; 
}




void EgammaHLTNxNClusterProducer::makeNxNClusters(edm::Event &evt, const edm::EventSetup &es,
						  const EcalRecHitCollection *hits, const reco::CaloID::Detectors detector)
{
  
  
  ///get status from DB
  edm::ESHandle<EcalChannelStatus> csHandle;
  if ( useDBStatus_ ) es.get<EcalChannelStatusRcd>().get(csHandle);
  const EcalChannelStatus &channelStatus = *csHandle; 
    
  
  std::vector<EcalRecHit> seeds;
  
  double clusterSeedThreshold ; 
  if (detector == reco::CaloID::DET_ECAL_BARREL){
    clusterSeedThreshold = clusSeedThr_;
  }else{
    clusterSeedThreshold = clusSeedThrEndCap_; 
  }
  

  for(EcalRecHitCollection::const_iterator itt = hits->begin(); itt != hits->end(); itt++){
    double energy = itt->energy();
    if( ! checkStatusOfEcalRecHit(channelStatus, *itt) ) continue; 
    if (energy > clusterSeedThreshold ) seeds.push_back(*itt);
    
    if( int(seeds.size()) > maxNumberofSeeds_){ //too many seeds, like beam splash events
      seeds.clear();  //// empty seeds vector, don't do clustering anymore
      break; 
    }
  }
  
  // get the geometry and topology from the event setup:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);
  
  const CaloSubdetectorGeometry *geometry_p;
  std::unique_ptr<CaloSubdetectorTopology> topology_p;
  if (detector == reco::CaloID::DET_ECAL_BARREL) {
    geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    topology_p.reset(new EcalBarrelTopology(geoHandle));
  }else {
    geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    topology_p.reset(new EcalEndcapTopology(geoHandle)); 
  }
  
  const CaloSubdetectorGeometry *geometryES_p;
  geometryES_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
    

  
  std::vector<reco::BasicCluster> clusters;
  std::vector<DetId> usedXtals;
  
  // sort seed according to Energy
  sort(seeds.begin(), seeds.end(), ecalRecHitSort());
  
  
  
  for (std::vector<EcalRecHit>::iterator itseed=seeds.begin(); itseed!=seeds.end(); itseed++) {
    DetId seed_id = itseed->id();
    std::vector<DetId>::const_iterator usedIds;
    
    std::vector<DetId>::iterator  itdet = find(usedXtals.begin(),usedXtals.end(),seed_id);
    if(itdet != usedXtals.end()) continue; 
    
    std::vector<DetId> clus_v = topology_p->getWindow(seed_id,clusEtaSize_,clusPhiSize_);	
    std::vector<std::pair<DetId, float> > clus_used;
    
    float clus_energy = 0; 
    
    for (std::vector<DetId>::iterator det=clus_v.begin(); det!=clus_v.end(); det++) {
	DetId detid = *det;
	
	//not yet used 
	std::vector<DetId>::iterator itdet = find(usedXtals.begin(),usedXtals.end(),detid);
	if(itdet != usedXtals.end()) continue; 
	//inside the collection
	EcalRecHitCollection::const_iterator hit  = hits->find(detid); 
	if( hit == hits->end()) continue; 
	
	if( ! checkStatusOfEcalRecHit(channelStatus, *hit) ) continue; 
	
	usedXtals.push_back(detid);
	clus_used.push_back(std::pair<DetId, float>(detid, 1.) );
	clus_energy += hit->energy();
	
    } //// end of making one nxn simple cluster
    
    if( clus_energy <= 0 ) continue; 
    
    math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_used,hits,geometry_p,geometryES_p);
    
    if (debug_>=2 ) LogDebug("")<<"nxn_cluster in run "<< evt.id().run()<<" event "<<evt.id().event()<<" energy: "<<clus_energy <<" eta: "<< clus_pos.Eta()<<" phi: "<< clus_pos.Phi()<<" nRecHits: "<< clus_used.size() <<std::endl;
    
    clusters.push_back(reco::BasicCluster(clus_energy, clus_pos, reco::CaloID(detector), clus_used, reco::CaloCluster::island, seed_id));
    if( int(clusters.size()) > maxNumberofClusters_){ ///if too much clusters made, then return 0 also 
      clusters.clear(); 
      break; 
    }
    
  }
  
  
  //Create empty output collections
  std::auto_ptr< reco::BasicClusterCollection > clusters_p(new reco::BasicClusterCollection);
  clusters_p->assign(clusters.begin(), clusters.end());
  if (detector == reco::CaloID::DET_ECAL_BARREL){
    if(debug_>=1) LogDebug("")<<"nxnclusterProducer: "<<clusters_p->size() <<" made in barrel"<<std::endl;
    evt.put(clusters_p, barrelClusterCollection_);
  }
  else {
    if(debug_>=1) LogDebug("")<<"nxnclusterProducer: "<<clusters_p->size() <<" made in endcap"<<std::endl;
    evt.put(clusters_p, endcapClusterCollection_);
  }
  
}

 
