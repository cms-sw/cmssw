#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterProducer.h"

#include "RecoParticleFlow/PFClusterAlgo/interface/PFClusterAlgo.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

// #include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
// #include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloTowerNavigator.h"


using namespace std;
using namespace edm;

PFClusterProducer::PFClusterProducer(const edm::ParameterSet& iConfig)
{
  


  processEcal_ = 
    iConfig.getUntrackedParameter<bool>("process_Ecal",true);
  processHcal_ = 
    iConfig.getUntrackedParameter<bool>("process_Hcal",true);
  processPS_ = 
    iConfig.getUntrackedParameter<bool>("process_PS",true);


  clusteringEcal_ = 
    iConfig.getUntrackedParameter<bool>("clustering_Ecal",true);
  clusteringHcal_ = 
    iConfig.getUntrackedParameter<bool>("clustering_Hcal",true);
  clusteringHcalCaloTowers_ = 
    iConfig.getUntrackedParameter<bool>("clustering_Hcal_CaloTowers",true);
  clusteringPS_ = 
    iConfig.getUntrackedParameter<bool>("clustering_PS",true);
    


  // parameters for ecal clustering
  
  threshEcalBarrel_ = 
    iConfig.getParameter<double>("thresh_Ecal_Barrel");
  threshSeedEcalBarrel_ = 
    iConfig.getParameter<double>("thresh_Seed_Ecal_Barrel");

  threshEcalEndcap_ = 
    iConfig.getParameter<double>("thresh_Ecal_Endcap");
  threshSeedEcalEndcap_ = 
    iConfig.getParameter<double>("thresh_Seed_Ecal_Endcap");


  nNeighboursEcal_ = 
    iConfig.getParameter<int>("nNeighbours_Ecal");

  posCalcP1Ecal_ = 
    iConfig.getParameter<double>("posCalcP1_Ecal");

  posCalcNCrystalEcal_ = 
    iConfig.getParameter<int>("posCalcNCrystal_Ecal");
    
  showerSigmaEcal_ = 
    iConfig.getParameter<double>("showerSigma_Ecal");
    

  

  // parameters for preshower clustering 

  threshPS_ = 
    iConfig.getParameter<double>("thresh_PS");
  threshSeedPS_ = 
    iConfig.getParameter<double>("thresh_Seed_PS");
  

  nNeighboursPS_ = 
    iConfig.getParameter<int>("nNeighbours_PS");

  posCalcP1PS_ = 
    iConfig.getParameter<double>("posCalcP1_PS");

  posCalcNCrystalPS_ = 
    iConfig.getParameter<int>("posCalcNCrystal_PS");
    
  showerSigmaPS_ = 
    iConfig.getParameter<double>("showerSigma_PS");
    

  // parameters for hcal clustering

  threshHcalBarrel_ = 
    iConfig.getParameter<double>("thresh_Hcal_Barrel");
  threshSeedHcalBarrel_ = 
    iConfig.getParameter<double>("thresh_Seed_Hcal_Barrel");

  threshHcalEndcap_ = 
    iConfig.getParameter<double>("thresh_Hcal_Endcap");
  threshSeedHcalEndcap_ = 
    iConfig.getParameter<double>("thresh_Seed_Hcal_Endcap");



  nNeighboursHcal_ = 
    iConfig.getParameter<int>("nNeighbours_Hcal");

  posCalcP1Hcal_ = 
    iConfig.getParameter<double>("posCalcP1_Hcal");

  posCalcNCrystalHcal_ = 
    iConfig.getParameter<int>("posCalcNCrystal_Hcal");
    
  showerSigmaHcal_ = 
    iConfig.getParameter<double>("showerSigma_Hcal");
  


  int    dcormode = 
    iConfig.getParameter<int>("depthCor_Mode");
  
  double dcora = 
    iConfig.getParameter<double>("depthCor_A");
  double dcorb = 
    iConfig.getParameter<double>("depthCor_B");
  double dcorap = 
    iConfig.getParameter<double>("depthCor_A_preshower");
  double dcorbp = 
    iConfig.getParameter<double>("depthCor_B_preshower");

  if( dcormode > -0.5 && 
      dcora > -0.5 && 
      dcorb > -0.5 && 
      dcorap > -0.5 && 
      dcorbp > -0.5 )
    reco::PFCluster::setDepthCorParameters( dcormode, 
					    dcora, dcorb, 
					    dcorap, dcorbp );

  
  ecalRecHitsEBModuleLabel_ = 
    iConfig.getUntrackedParameter<string>("ecalRecHitsEBModuleLabel",
					  "ecalRecHit");
  ecalRecHitsEBProductInstanceName_ = 
    iConfig.getUntrackedParameter<string>("ecalRecHitsEBProductInstanceName",
					  "EcalRecHitsEB");
  
  ecalRecHitsEEModuleLabel_ = 
    iConfig.getUntrackedParameter<string>("ecalRecHitsEEModuleLabel",
					  "ecalRecHit");
  ecalRecHitsEEProductInstanceName_ = 
    iConfig.getUntrackedParameter<string>("ecalRecHitsEEProductInstanceName",
					  "EcalRecHitsEE");
  
  ecalRecHitsESModuleLabel_ = 
    iConfig.getUntrackedParameter<string>("ecalRecHitsESModuleLabel",
					  "ecalRecHit");
  ecalRecHitsESProductInstanceName_ = 
    iConfig.getUntrackedParameter<string>("ecalRecHitsESProductInstanceName",
					  "EcalRecHitsES");
  
  
  hcalRecHitsHBHEModuleLabel_ = 
    iConfig.getUntrackedParameter<string>("hcalRecHitsHBHEModuleLabel",
					  "hbhereco");
  hcalRecHitsHBHEProductInstanceName_ = 
    iConfig.getUntrackedParameter<string>("hcalRecHitsHBHEProductInstanceName",
					  "");
    
  caloTowersModuleLabel_ = 
    iConfig.getUntrackedParameter<string>("caloTowersModuleLabel",
					  "towerMaker");
  caloTowersProductInstanceName_ = 
    iConfig.getUntrackedParameter<string>("caloTowersProductInstanceName",
					  "");
    
//   produceClusters_ = 
//     iConfig.getUntrackedParameter<bool>("produceClusters", true );
    

  produceRecHits_ = 
    iConfig.getUntrackedParameter<bool>("produce_RecHits", false );

  //register products
  if(produceRecHits_) {
    produces<reco::PFRecHitCollection>("ECAL");
    produces<reco::PFRecHitCollection>("HCAL");
    produces<reco::PFRecHitCollection>("PS");
  }
  produces<reco::PFClusterCollection>("ECAL");
  produces<reco::PFClusterCollection>("HCAL");
  produces<reco::PFClusterCollection>("PS");
}



PFClusterProducer::~PFClusterProducer() {}




void PFClusterProducer::produce(edm::Event& iEvent, 
				const edm::EventSetup& iSetup) {
  
  if( processEcal_ ) produceEcal(iEvent, iSetup);
  if( processHcal_ ) produceHcal(iEvent, iSetup);
  if(processPS_) producePS(iEvent, iSetup);
}



void PFClusterProducer::produceEcal( edm::Event& iEvent, 
				     const edm::EventSetup& iSetup ) {

  map<unsigned,  reco::PFRecHit* > idSortedRecHits;
  
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<IdealGeometryRecord>().get(geoHandle);
  
  // get the ecalBarrel geometry
  const CaloSubdetectorGeometry *ecalBarrelGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  
  // get the ecalBarrel topology
  EcalBarrelTopology ecalBarrelTopology(geoHandle);
  
  // get the endcap geometry
  const CaloSubdetectorGeometry *ecalEndcapGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

  // get the endcap topology
  EcalEndcapTopology ecalEndcapTopology(geoHandle);


         
  // get the ecalBarrel rechits

  edm::Handle<EcalRecHitCollection> rhcHandle;
  try {
    iEvent.getByLabel(ecalRecHitsEBModuleLabel_, 
		      ecalRecHitsEBProductInstanceName_, 
		      rhcHandle);
    if (!(rhcHandle.isValid())) {
      edm::LogError("PFClusterProducer")
	<<"could not get a handle on EcalRecHitsEB!"<<endl;
      return;
    }
      
    // process ecal ecalBarrel rechits
    for(unsigned i=0; i<rhcHandle->size(); i++) {
      
      const EcalRecHit& erh = (*rhcHandle)[i];
      const DetId& detid = erh.detid();
      double energy = erh.energy();

      if(energy < threshEcalBarrel_ ) continue;


      reco::PFRecHit *pfrh = createEcalRecHit(detid, energy,  
					      PFLayer::ECAL_BARREL,
					      ecalBarrelGeometry);

      if( !pfrh ) continue; // problem with this rechit. skip it

      idSortedRecHits.insert( make_pair(detid.rawId(), pfrh) ); 
    }      
  }
  catch ( cms::Exception& ex ) {
    edm::LogError("PFClusterProducerError")
      <<"Error! can't get the ecal barrel rechits "<<ex.what()<<endl;
    return;
  }



  // process ecal endcap rechits

  try {
    iEvent.getByLabel(ecalRecHitsEEModuleLabel_,
		      ecalRecHitsEEProductInstanceName_,
		      rhcHandle);
    if (!(rhcHandle.isValid())) {
      LogError("PFClusterProducer")
	<<"could not get a handle on EcalRecHitsEE!"<<endl;
      return;
    }
    
    // cout<<"process endcap rechits"<<endl;
    for(unsigned i=0; i<rhcHandle->size(); i++) {
      
      const EcalRecHit& erh = (*rhcHandle)[i];
      const DetId& detid = erh.detid();
      double energy = erh.energy();
     
      if(energy < threshEcalEndcap_ ) continue;

      reco::PFRecHit *pfrh = createEcalRecHit(detid, energy,
					      PFLayer::ECAL_ENDCAP,
					      ecalEndcapGeometry);
      
      if( !pfrh ) continue; // problem with this rechit. skip it

      idSortedRecHits.insert( make_pair(detid.rawId(), pfrh) ); 
    }
  }
  catch ( cms::Exception& ex ) {
    edm::LogError("PFClusterProducerError")
      <<"Error! can't get the EE rechits "<<ex.what()<<endl;
    return;
  }




  // find rechits neighbours
  for( PFClusterAlgo::IDH ih = idSortedRecHits.begin(); 
       ih != idSortedRecHits.end(); ih++) {
    findRecHitNeighbours( ih->second, idSortedRecHits, 
			  ecalBarrelTopology, 
			  *ecalBarrelGeometry, 
			  ecalEndcapTopology,
			  *ecalEndcapGeometry);
  }
    
  if(clusteringEcal_) {
    PFClusterAlgo clusteralgo; 
      
    clusteralgo.setThreshBarrel( threshEcalBarrel_ );
    clusteralgo.setThreshSeedBarrel( threshSeedEcalBarrel_ );
      
    clusteralgo.setThreshEndcap( threshEcalEndcap_ );
    clusteralgo.setThreshSeedEndcap( threshSeedEcalEndcap_ );
        
    clusteralgo.setNNeighbours( nNeighboursEcal_ );
    clusteralgo.setPosCalcNCrystal( posCalcNCrystalEcal_ );
    clusteralgo.setPosCalcP1( posCalcP1Ecal_ );
    clusteralgo.setShowerSigma( showerSigmaEcal_ );
      

    clusteralgo.init( idSortedRecHits ); 
    clusteralgo.doClustering();

    LogInfo("PFClusterProducer")
      <<" ECAL clusters --------------------------------- "<<endl
      <<clusteralgo<<endl;

    // get clusters out of the clustering algorithm 
    // and put them in the event. There is no copy.
    auto_ptr< vector<reco::PFCluster> > 
      outClustersECAL( clusteralgo.clusters() ); 
    iEvent.put( outClustersECAL, "ECAL");
  }    


  // if requested, get rechits passing the threshold from algo, 
  // and pass them to the event.
  if(produceRecHits_) {
      

    auto_ptr< vector<reco::PFRecHit> > 
      recHits( new vector<reco::PFRecHit> ); 
      
    recHits->reserve( idSortedRecHits.size() ); 
      
    for( PFClusterAlgo::IDH ih = idSortedRecHits.begin(); 
	 ih != idSortedRecHits.end(); ih++) {  
      recHits->push_back( reco::PFRecHit( *(ih->second) ) );    
    }
     
    iEvent.put( recHits, "ECAL" );
  }
    
    

  // clear all rechits
  for( PFClusterAlgo::IDH ih = idSortedRecHits.begin(); 
       ih != idSortedRecHits.end(); ih++) {  
    delete ih->second;
  }

}


void PFClusterProducer::produceHcal(edm::Event& iEvent, 
				    const edm::EventSetup& iSetup) {


  map<unsigned,  reco::PFRecHit* > hcalrechits;

  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<IdealGeometryRecord>().get(geoHandle);
    
  // get the hcalBarrel geometry
  const CaloSubdetectorGeometry *hcalBarrelGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);

  // get the endcap geometry
  const CaloSubdetectorGeometry *hcalEndcapGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

 
  // 2 possibilities to make HCAL clustering :
  // - from the HCAL rechits
  // - from the CaloTowers. 
  // ultimately, clustering will be done taking CaloTowers as an 
  // input. This possibility is currently under investigation, and 
  // was thus made optional.

  // in the first step, we will fill the map of PFRecHits hcalrechits
  // either from CaloTowers or from HCAL rechits. 

  // in the second step, we will perform clustering on this map.

  if( clusteringHcalCaloTowers_ ) {
      
    edm::Handle<CaloTowerCollection> caloTowers; 
    CaloTowerTopology caloTowerTopology; 
    const CaloSubdetectorGeometry *caloTowerGeometry = 0; 
    // = geometry_->getSubdetectorGeometry(id)

    try {
      // get calotowers
      iEvent.getByLabel(caloTowersModuleLabel_,
			caloTowersProductInstanceName_,
			caloTowers);

      // create rechits
      typedef CaloTowerCollection::const_iterator ICT;

      for(ICT ict=caloTowers->begin(); ict!=caloTowers->end();ict++) {

	const CaloTower& ct = (*ict);
	
	if(!caloTowerGeometry) 
	  caloTowerGeometry = geoHandle->getSubdetectorGeometry(ct.id());
	  
	// get the hadronic energy.
	double energy = ct.hadEnergy();
	if( energy < 1e-9 ) continue;  
	 
	

	// the layer will be taken from the first constituent. 
	// all thresholds for ECAL must be set to very high values !!!
	assert( ct.constituentsSize() );	  
	const HcalDetId& detid = ct.constituent(0);
	  
	int layer = 0; 

	switch( detid.subdet() ) {
	case HcalBarrel:
	  if(energy > threshHcalBarrel_)
	    layer = PFLayer::HCAL_BARREL1;
	  break;
	case HcalEndcap:
	  if(energy > threshHcalEndcap_)
	    layer = PFLayer::HCAL_ENDCAP;
	  break;
	default:
	  LogError("PFClusterProducer")
	    <<"CaloTower constituent: unknown layer : "
	    <<detid.subdet()<<endl;
	} 
	  
	// either detid.subdet() not treated in switch/case
	// or calotower under threshold
	if(!layer) continue; 


	// get the geometry 

	const CaloCellGeometry* geom = geoHandle->getGeometry( ct.id() );
	if(!geom) {
	  edm::LogError("PFClusterProducer")
	    <<"warning CaloTower detid "
	    <<ct.id().rawId()<<" not found in CaloGeometry"<<endl;
	  continue;
	}

	const GlobalPoint& position = geom->getPosition();


	reco::PFRecHit* pfrechit 
	  = new reco::PFRecHit( ct.id().rawId(), layer, energy, 
				position.x(), position.y(), position.z(), 
				0,0,0 );

	// note that the key is the CaloTower detid, 
	// not the constituent detid. 	  
	hcalrechits.insert( make_pair(ct.id().rawId(), pfrechit) ); 
      }


      // do navigation 
      for( PFClusterAlgo::IDH ih = hcalrechits.begin(); 
	   ih != hcalrechits.end(); ih++) {
	findRecHitNeighboursCT( ih->second, 
				hcalrechits, 
				caloTowerTopology, 
				*caloTowerGeometry );
      }
    }
    catch ( cms::Exception& ex ) {
      edm::LogError("PFClusterProducerError")
	<<"Error! can't get the CaloTowers "<<ex.what()<<endl;
      return;
    }
  }
  else { // clustering is not done on CaloTowers but on HCAL rechits.
       

    // get the hcal topology
    HcalTopology hcalTopology;
    
    // HCAL rechits 
    vector<edm::Handle<HBHERecHitCollection> > hcalHandles;  
    try {

      // retry this:
      //       iEvent.getByLabel(hcalRecHitsHBHEModuleLabel_,
      // 			hcalRecHitsHBHEProductInstanceName_,
      // 			hcalHandles);
      // instead of this:
      iEvent.getManyByType(hcalHandles);      
    
      for(unsigned ih=0; ih<hcalHandles.size(); ih++) {
	const edm::Handle<HBHERecHitCollection>& handle = hcalHandles[ih];
      
	for(unsigned irechit=0; irechit<handle->size(); irechit++) {
	  const HBHERecHit& hit = (*handle)[irechit];
	
	  double energy = hit.energy();
	
	  reco::PFRecHit* pfrechit = 0;
	  
	  const HcalDetId& detid = hit.detid();
	  switch( detid.subdet() ) {
	  case HcalBarrel:
	    if(energy > threshHcalBarrel_){
	      pfrechit = createHcalRecHit(detid, 
					  energy, 
					  PFLayer::HCAL_BARREL1, 
					  hcalBarrelGeometry );
	    }
	    break;
	  case HcalEndcap:
	    if(energy > threshHcalEndcap_){
	      pfrechit = createHcalRecHit(detid, 
					  energy, 
					  PFLayer::HCAL_ENDCAP, 
					  hcalEndcapGeometry );
	    }
	    break;
	  default:
	    LogError("PFClusterProducer")
	      <<"HCAL rechit: unknown layer : "<<detid.subdet()<<endl;
	    continue;
	  } 

	  if(pfrechit) 	    
	    hcalrechits.insert( make_pair(detid.rawId(), pfrechit) ); 
	  
	}
  

	// do navigation:
	for( PFClusterAlgo::IDH ih = hcalrechits.begin(); 
	     ih != hcalrechits.end(); ih++) {
	  findRecHitNeighbours( ih->second, hcalrechits, 
				hcalTopology, 
				*hcalBarrelGeometry, 
				hcalTopology,
				*hcalEndcapGeometry);
	}
      }      
    } catch (...) {
      LogError("PFClusterProducer")
	<<"could not get handles on HBHERecHits!"<< endl;
      return;
    }
  }



  if(clusteringHcal_) {

    PFClusterAlgo clusteralgo; 
	  
    clusteralgo.setThreshBarrel( threshHcalBarrel_ );
    clusteralgo.setThreshSeedBarrel( threshSeedHcalBarrel_ );
	  
    clusteralgo.setThreshEndcap( threshHcalEndcap_ );
    clusteralgo.setThreshSeedEndcap( threshSeedHcalEndcap_ );
    
    clusteralgo.setNNeighbours( nNeighboursHcal_ );
    clusteralgo.setPosCalcNCrystal( posCalcNCrystalHcal_ );
    clusteralgo.setPosCalcP1( posCalcP1Hcal_ );
    clusteralgo.setShowerSigma( showerSigmaHcal_ );

    clusteralgo.init( hcalrechits ); 
    clusteralgo.doClustering();
	
    LogInfo("PFClusterProducer")
      <<" HCAL clusters --------------------------------- "<<endl
      <<clusteralgo<<endl;
	
    auto_ptr< vector<reco::PFCluster> > 
      outClustersHCAL( clusteralgo.clusters() ); 
    // 	outClustersHCAL = clusteralgo.clusters();
    iEvent.put( outClustersHCAL, "HCAL");	  
  }


  // if requested, get rechits passing the threshold from algo, 
  // and pass them to the event.
  if(produceRecHits_) {
	  
    auto_ptr< vector<reco::PFRecHit> > 
      recHits( new vector<reco::PFRecHit> ); 
    recHits->reserve( hcalrechits.size() );
	  
    for( PFClusterAlgo::IDH ih = hcalrechits.begin(); 
	 ih != hcalrechits.end(); ih++) {
      recHits->push_back( reco::PFRecHit( *(ih->second) ) );    
    }
	  
    iEvent.put( recHits, "HCAL" );
  }

	
  // clear all rechits
  for( PFClusterAlgo::IDH ih = hcalrechits.begin(); 
       ih != hcalrechits.end(); ih++) {
	
    delete ih->second;
  }
}



void PFClusterProducer::producePS(edm::Event& iEvent, 
				  const edm::EventSetup& iSetup) {
  
  map<unsigned,  reco::PFRecHit* > psrechits;

  // get the ps geometry
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<IdealGeometryRecord>().get(geoHandle);
    
  const CaloSubdetectorGeometry *psGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
    
  // get the ps topology
  EcalPreshowerTopology psTopology(geoHandle);

  // process rechits
  Handle< EcalRecHitCollection >   pRecHits;


  try {
    iEvent.getByLabel(ecalRecHitsESModuleLabel_,
		      ecalRecHitsESProductInstanceName_,
		      pRecHits);
    if (!(pRecHits.isValid())) {
      LogError("PFClusterProducer")
	<<"could not get a handle on preshower rechits!"<<endl;
      return;
    }

    const EcalRecHitCollection& rechits = *( pRecHits.product() );
    typedef EcalRecHitCollection::const_iterator IT;
 
    for(IT i=rechits.begin(); i!=rechits.end(); i++) {
      const EcalRecHit& hit = *i;
      
      double energy = hit.energy();
      if( energy < threshPS_ ) continue; 
            
      const ESDetId& detid = hit.detid();
      const CaloCellGeometry *thisCell = psGeometry->getGeometry(detid);
     
      if(!thisCell) {
	LogError("PFClusterProducer")<<"warning detid "<<detid.rawId()
				     <<" not found in preshower geometry"
				     <<endl;
	return;
      }
      
      const GlobalPoint& position = thisCell->getPosition();
     
      int layer = 0;
            
      switch( detid.plane() ) {
      case 1:
	layer = PFLayer::PS1;
	break;
      case 2:
	layer = PFLayer::PS2;
	break;
      default:
	LogError("PFClusterProducer")
	  <<"incorrect preshower plane !! plane number "
	  <<detid.plane()<<endl;
	assert(0);
      }
 
      reco::PFRecHit *pfrechit 
	= new reco::PFRecHit( detid.rawId(), layer, energy, 
			      position.x(), position.y(), position.z(), 
			      0,0,0 );
	
      psrechits.insert( make_pair(detid.rawId(), pfrechit) );
	
    }
    
    // cout<<"find rechits neighbours"<<endl;
    for( PFClusterAlgo::IDH ih = psrechits.begin(); 
	 ih != psrechits.end(); ih++) {
	
      findRecHitNeighbours( ih->second, psrechits, 
			    psTopology, 
			    *psGeometry, 
			    psTopology,
			    *psGeometry);
    }
      
    if(clusteringPS_) {

      PFClusterAlgo clusteralgo; 
	
      clusteralgo.setThreshEndcap( threshPS_ );
      clusteralgo.setThreshSeedEndcap( threshSeedPS_ );

      clusteralgo.setNNeighbours( nNeighboursPS_ );
      clusteralgo.setPosCalcNCrystal( posCalcNCrystalPS_ );
      clusteralgo.setPosCalcP1( posCalcP1PS_ );
      clusteralgo.setShowerSigma( showerSigmaPS_ );
	
      clusteralgo.init( psrechits ); 
      clusteralgo.doClustering();

      LogInfo("PFClusterProducer")
	<<" Preshower clusters --------------------------------- "<<endl
	<<clusteralgo<<endl;
	
      auto_ptr< vector<reco::PFCluster> > 
	outClustersPS( clusteralgo.clusters() ); 
      iEvent.put( outClustersPS, "PS");
    }    

    // if requested, get rechits passing the threshold from algo, 
    // and pass them to the event.
    if(produceRecHits_) {
	
      auto_ptr< vector<reco::PFRecHit> > 
	recHits( new vector<reco::PFRecHit> ); 
      recHits->reserve( psrechits.size() );
	
      for( PFClusterAlgo::IDH ih = psrechits.begin(); 
	   ih != psrechits.end(); ih++) {  
	recHits->push_back( reco::PFRecHit( *(ih->second) ) );    
      }
	
      iEvent.put( recHits, "PS" );
    }
      

    // clear all 
    for( PFClusterAlgo::IDH ih = psrechits.begin(); 
	 ih != psrechits.end(); ih++) {  
      delete ih->second;
    }
  }
  catch ( cms::Exception& ex ) {
    edm::LogError("PFClusterProducer") 
      <<"Error! can't get the preshower rechits. module: "
      <<ecalRecHitsESModuleLabel_
      <<", product instance: "<<ecalRecHitsESProductInstanceName_
      <<endl;
  }
}



reco::PFRecHit* 
PFClusterProducer::createEcalRecHit( const DetId& detid,
				     double energy,
				     int layer,
				     const CaloSubdetectorGeometry* geom ) {

  math::XYZVector position;
  math::XYZVector axis;
  bool geomfound = findEcalRecHitGeometry( detid, geom,
					   position, axis ); 
  
  if( !geomfound ) {
    LogError("PFClusterProducer")<<"cannor find geometry for detid "
				 <<detid.rawId()<<" in layer "<<layer<<endl;
    return 0; // geometry not found, skip rechit
  }
  
  
  reco::PFRecHit *pfrh 
    = new reco::PFRecHit( detid.rawId(), layer, 
			  energy, 
			  position.x(), position.y(), position.z(), 
			  axis.x(), axis.y(), axis.z() ); 

  return pfrh;
}





reco::PFRecHit* 
PFClusterProducer::createHcalRecHit( const DetId& detid,
				     double energy,
				     int layer,
				     const CaloSubdetectorGeometry* geom ) {
  
  const CaloCellGeometry *thisCell = geom->getGeometry(detid);
  if(!thisCell) {
    edm::LogError("PFClusterProducer")
      <<"warning detid "<<detid.rawId()<<" not found in layer "
      <<layer<<endl;
    return 0;
  }
  
  const GlobalPoint& position = thisCell->getPosition();
  
  reco::PFRecHit *rh = 
    new reco::PFRecHit( detid.rawId(),  layer, energy, 
			position.x(), position.y(), position.z(), 
			0,0,0 );
  
  return rh;
}




// reco::PFRecHit* 
// PFClusterProducer::createHcalCTRecHit( const DetId& detid,
// 				       double energy,
// 				       int layer,
// 				       const CaloSubdetectorGeometry* geom ) {
  
//   const CaloCellGeometry *thisCell = geom->getGeometry(detid);
//   if(!thisCell) {
//     edm::LogError("PFClusterProducer")
//       <<"warning detid "<<detid.rawId()<<" not found in layer "
//       <<layer<<endl;
//     return 0;
//   }
  
//   const GlobalPoint& position = thisCell->getPosition();
  
//   reco::PFRecHit *rh = 
//     new reco::PFRecHit( detid.rawId(),  layer, energy, 
// 			position.x(), position.y(), position.z(), 
// 			0,0,0 );
  
//   return rh;
// }






bool
PFClusterProducer::findEcalRecHitGeometry(const DetId& detid, 
					  const CaloSubdetectorGeometry* geom,
					  math::XYZVector& position, 
					  math::XYZVector& axis ) {
  

  const CaloCellGeometry *thisCell 
    = geom->getGeometry(detid);
  
  // find rechit geometry
  if(!thisCell) {
    LogError("PFClusterProducer")
      <<"warning detid "<<detid.rawId()
      <<" not found in geometry"<<endl;
    return false;
  }
  
  position.SetCoordinates ( thisCell->getPosition().x(),
			    thisCell->getPosition().y(),
			    thisCell->getPosition().z() );

  
  
  // the axis vector is the difference 
  const TruncatedPyramid* pyr 
    = dynamic_cast< const TruncatedPyramid* > (thisCell);    
  if( pyr ) {
    axis.SetCoordinates( pyr->getPosition(1).x(), 
			 pyr->getPosition(1).y(), 
			 pyr->getPosition(1).z() ); 
    
    math::XYZVector axis0( pyr->getPosition(0).x(), 
			   pyr->getPosition(0).y(), 
			   pyr->getPosition(0).z() );
    
    axis -= axis0;

    
    return true;
  }
  else return false;
}


void 
PFClusterProducer::findRecHitNeighbours
( reco::PFRecHit* rh, 
  const map<unsigned, reco::PFRecHit* >& rechits, 
  const CaloSubdetectorTopology& barrelTopology, 
  const CaloSubdetectorGeometry& barrelGeometry, 
  const CaloSubdetectorTopology& endcapTopology, 
  const CaloSubdetectorGeometry& endcapGeometry ) {
  
  const math::XYZPoint& cpos = rh->positionXYZ();
  double posx = cpos.X();
  double posy = cpos.Y();
  double posz = cpos.Z();

  bool debug = false;
//   if( rh->layer() == PFLayer::PS1 ||
//       rh->layer() == PFLayer::PS2 ) debug = true;
  

  DetId detid( rh->detId() );

  CaloNavigator<DetId>* navigator = 0;
  CaloSubdetectorGeometry* geometry = 0;
  CaloSubdetectorGeometry* othergeometry = 0;



  if(debug) cerr<<"find hcal neighbours "<<rh->layer()<<endl;
  
  switch( rh->layer()  ) {
  case PFLayer::ECAL_ENDCAP: 
    // if(debug) cerr<<"ec cell"<<endl;
    navigator = new CaloNavigator<DetId>(detid, &endcapTopology);
    geometry = const_cast< CaloSubdetectorGeometry* > (&endcapGeometry);
    break;
  case PFLayer::ECAL_BARREL: 
    navigator = new CaloNavigator<DetId>(detid, &barrelTopology);
    geometry = const_cast< CaloSubdetectorGeometry* > (&barrelGeometry);
    break;
  case PFLayer::HCAL_ENDCAP:
    navigator = new CaloNavigator<DetId>(detid, &endcapTopology);
    geometry = const_cast< CaloSubdetectorGeometry* > (&endcapGeometry);
    othergeometry 
      = const_cast< CaloSubdetectorGeometry* > (&barrelGeometry);
    break;
  case PFLayer::HCAL_BARREL1:
    navigator = new CaloNavigator<DetId>(detid, &barrelTopology);
    geometry = const_cast< CaloSubdetectorGeometry* > (&barrelGeometry);
    othergeometry 
      = const_cast< CaloSubdetectorGeometry* > (&endcapGeometry);
    break;
  case PFLayer::PS1:
  case PFLayer::PS2:
    navigator = new CaloNavigator<DetId>(detid, &barrelTopology);
    geometry = const_cast< CaloSubdetectorGeometry* > (&barrelGeometry);
    othergeometry 
      = const_cast< CaloSubdetectorGeometry* > (&endcapGeometry);
    break;
  default:
    assert(0);
  }

  assert( navigator && geometry );

  // if(debug) cerr<<"nav and geom are non 0"<<endl;

  if(debug) cerr<<"calling north"<<endl;
  DetId north = navigator->north();  
  if(debug) cerr<<"north"<<endl;
  
  DetId northeast(0);
  if( north != DetId(0) ) {
    northeast = navigator->east();  
    if( northeast != DetId(0) ) {

//       ESDetId esid(northeast.rawId());
//       cout<<"nb layer : "<<esid.plane()<<endl;

      const CaloCellGeometry * nbcell = geometry->getGeometry(northeast);
      if(!nbcell)
	nbcell = othergeometry->getGeometry(northeast);

      if(nbcell) {
	const GlobalPoint& nbpos = nbcell->getPosition();
	double cposx = nbpos.x();
	cposx += posx; 
	cposx /= 2.;
	double cposy = nbpos.y();
	cposy += posy; 
	cposy /= 2.;
	double cposz = nbpos.z();
	cposz += posz; 
	cposz /= 2.;
	
	rh->setNECorner( cposx, cposy, cposz );
      }
      else if(debug) 
	cerr<<cpos.Eta()<<" "<<cpos.Phi()
	    <<"geometry not found for detid "<<northeast.rawId()
	    <<" NE corner not set"<<endl;
    }
    else if(debug) cerr<<cpos.Eta()<<" "<<cpos.Phi()
			 <<"invalid detid NE corner not set"<<endl;
  }
  navigator->home();

  if(debug) cerr<<"ne ok"<<endl;

  DetId south = navigator->south();

  if(debug) cerr<<"south ok"<<endl;
  

  DetId southwest(0); 
  if( south != DetId(0) ) {
  
    southwest = navigator->west();
    if(debug) cerr<<1<<endl;
    if( southwest != DetId(0) ) {
      if(debug) cerr<<2<<endl;
      const CaloCellGeometry * nbcell = geometry->getGeometry(southwest);

      // now that we have moved, it could be that the neighbour is not in 
      // the same subdetector. 
      // the other geometry is hence used
      if(!nbcell)
	nbcell = othergeometry->getGeometry(southwest);

      if(nbcell) {
	
	const GlobalPoint& nbpos = nbcell->getPosition();
	double cposx = nbpos.x();
	cposx += posx; 
	cposx /= 2.;
	double cposy = nbpos.y();
	cposy += posy; 
	cposy /= 2.;
	double cposz = nbpos.z();
	cposz += posz; 
	cposz /= 2.;
	
	rh->setSWCorner( cposx, cposy, cposz );
      }
      else if(debug) 
	cerr<<cpos.Eta()<<" "<<cpos.Phi()
	    <<"geometry not found for detid "<<southwest.rawId()
	    <<" SW corner not set"<<endl;
    }
    else if(debug) cerr<<cpos.Eta()<<" "<<cpos.Phi()
			 <<"invalid detid SW corner not set"<<endl;
  }
  navigator->home();
  if(debug) cerr<<"sw ok"<<endl;


  DetId east = navigator->east();
  if(debug) cerr<<"e ok"<<endl;
  DetId southeast;
  if( east != DetId(0) ) {
    southeast = navigator->south(); 
    if( southeast != DetId(0) ) {
      const CaloCellGeometry * nbcell = geometry->getGeometry(southeast);
      if(!nbcell) 
	nbcell = othergeometry->getGeometry(southeast);

      if(nbcell) {
	const GlobalPoint& nbpos = nbcell->getPosition();
	double cposx = nbpos.x();
	cposx += posx; 
	cposx /= 2.;
	double cposy = nbpos.y();
	cposy += posy; 
	cposy /= 2.;
	double cposz = nbpos.z();
	cposz += posz; 
	cposz /= 2.;
      
	rh->setSECorner( cposx, cposy, cposz );
      }
      else  if(debug) cerr<<cpos.Eta()<<" "<<cpos.Phi()
			  <<"geometry not found for detid "<<southeast.rawId()
			  <<" SE corner not set"<<endl;
    }
    else if(debug) cerr<<cpos.Eta()<<" "<<cpos.Phi()
			 <<"invalid detid SE corner not set"<<endl;
  }
  navigator->home();
  if(debug) cerr<<"se ok"<<endl;
  DetId west = navigator->west();
  if(debug) cerr<<"w ok"<<endl;
  DetId northwest;
  if( west != DetId(0) ) {   
    northwest = navigator->north();  
    if( northwest != DetId(0) ) {
      const CaloCellGeometry * nbcell = geometry->getGeometry(northwest);
      if(!nbcell) 
	nbcell = othergeometry->getGeometry(northwest);

      if(nbcell) {
	const GlobalPoint& nbpos = nbcell->getPosition();
	double cposx = nbpos.x();
	cposx += posx; 
	cposx /= 2.;
	double cposy = nbpos.y();
	cposy += posy; 
	cposy /= 2.;
	double cposz = nbpos.z();
	cposz += posz; 
	cposz /= 2.;
      
	rh->setNWCorner( cposx, cposy, cposz );
      }
      else if(debug) cerr<<cpos.Eta()<<" "<<cpos.Phi()
			 <<"geometry not found for detid "<<northwest.rawId()
			 <<" NW corner not set"<<endl;
    }
    else if(debug) cerr<<cpos.Eta()<<" "<<cpos.Phi()
			 <<"invalid detid NW corner not set"<<endl;
  }
  navigator->home();
  if(debug) cerr<<"nw ok"<<endl;
    
  reco::PFRecHit* rhnorth = 0;
  PFClusterAlgo::IDH i = rechits.find( north.rawId() );
  if(i != rechits.end() ) 
    rhnorth = i->second;
    
  reco::PFRecHit* rhnortheast = 0;
  i = rechits.find( northeast.rawId() );
  if(i != rechits.end() ) 
    rhnortheast = i->second;
    
  reco::PFRecHit* rhsouth = 0;
  i = rechits.find( south.rawId() );
  if(i != rechits.end() ) 
    rhsouth = i->second;
    
  reco::PFRecHit* rhsouthwest = 0;
  i = rechits.find( southwest.rawId() );
  if(i != rechits.end() ) 
    rhsouthwest = i->second;
    
  reco::PFRecHit* rheast = 0;
  i = rechits.find( east.rawId() );
  if(i != rechits.end() ) 
    rheast = i->second;
    
  reco::PFRecHit* rhsoutheast = 0;
  i = rechits.find( southeast.rawId() );
  if(i != rechits.end() ) 
    rhsoutheast = i->second;
    
  reco::PFRecHit* rhwest = 0;
  i = rechits.find( west.rawId() );
  if(i != rechits.end() ) 
    rhwest = i->second;
    
  reco::PFRecHit* rhnorthwest = 0;
  i = rechits.find( northwest.rawId() );
  if(i != rechits.end() ) 
    rhnorthwest = i->second;
    
  vector<reco::PFRecHit*> neighbours;
  neighbours.reserve(8);
  neighbours.push_back( rhnorth );
  neighbours.push_back( rhnorthwest );
  neighbours.push_back( rhwest );
  neighbours.push_back( rhsouthwest );
  neighbours.push_back( rhsouth );
  neighbours.push_back( rhsoutheast );
  neighbours.push_back( rheast );
  neighbours.push_back( rhnortheast );
    
  rh->setNeighbours( neighbours );

//   cout<<(*rh)<<endl;

}


void 
PFClusterProducer::findRecHitNeighboursCT
( reco::PFRecHit* rh, 
  const map<unsigned, reco::PFRecHit* >& rechits, 
  const CaloSubdetectorTopology& topology, 
  const CaloSubdetectorGeometry& geometry ) {
 
  const math::XYZVector& rhpos = math::XYZVector(rh->positionXYZ());
  double rhposx = rhpos.X();
  double rhposy = rhpos.Y();
  double rhposz = rhpos.Z();
  

  math::XYZVector dummy(rhpos);
  dummy += rhpos;

  bool debug = false;
  //   if( rh->layer() == PFLayer::PS1 ||
  //       rh->layer() == PFLayer::PS2 ) debug = true;
  
  
  DetId detid( rh->detId() );

  vector<DetId> northids = topology.north(detid);
  vector<DetId> westids = topology.west(detid);
  vector<DetId> southids = topology.south(detid);
  vector<DetId> eastids = topology.east(detid);

  DetId badId;

  DetId north;
  DetId northwest;
  DetId west;
  DetId southwest;
  DetId south;
  DetId southeast;
  DetId east;
  DetId northeast;
  
  // for north and south, there is no ambiguity : 1 or 0 neighbours
  string err("PFClusterProducer::findRecHitNeighboursCT : incorrect number of neighbours "); 
  char n[20];
  
  switch( northids.size() ) {
  case 0: 
    break;
  case 1: 
    north = northids[0];
    break;
  default:
    sprintf(n, "north: %d", northids.size() );
    err += n;
    throw( err ); 
  }

  switch( southids.size() ) {
  case 0: 
    break;
  case 1: 
    south = southids[0];
    break;
  default:
    sprintf(n, "south %d", southids.size() );
    err += n;
    throw( err ); 
  }
  
  // for east and west, one must take care 
  // of the pitch change in HCAL endcap.

  switch( eastids.size() ) {
  case 0: 
    break;
  case 1: 
    east = eastids[0];
    northeast = getNorth(east, topology);
    southeast = getSouth(east, topology);
    break;
  case 2:  
    // in this case, 0 is more on the north than 1
    east = eastids[0];
    northeast = getNorth(east, topology );
    southeast = getSouth(eastids[1], topology);    
    break;
  default:
    sprintf(n, "%d", eastids.size() );
    err += n;
    throw( err ); 
  }
  
  
  switch( westids.size() ) {
  case 0: 
    break;
  case 1: 
    west = westids[0];
    northwest = getNorth(west, topology);
    southwest = getSouth(west, topology);
    break;
  case 2:  
    // in this case, 0 is more on the north than 1
    west = westids[0];
    northwest = getNorth(west, topology );
    southwest = getSouth(westids[1], topology );    
    break;
  default:
    sprintf(n, "%d", westids.size() );
    err += n;
    throw( err ); 
  }


  
  // set corners
  // Attention !!! in CMSSW east corresponds to negative eta. 
  // this is counterintuitive -> changing it. 

  if(northeast != badId) {
    const CaloCellGeometry *thisCell = geometry.getGeometry(northeast);
    if(!thisCell) {
      LogError("PFClusterProducer")
	<<"warning detid "<<detid.rawId()
	<<" not found in geometry"<<endl;
    }
    else {
      const GlobalPoint& pos = thisCell->getPosition();
      double cornerposx = pos.x(); 
      double cornerposy = pos.y(); 
      double cornerposz = pos.z(); 
      cornerposx += rhposx; cornerposx /= 2.;
      cornerposy += rhposy; cornerposy /= 2.;
      cornerposz += rhposz; cornerposz /= 2.;
            
      rh->setNWCorner(cornerposx, cornerposy, cornerposz);
    }
  }
 

  if(southeast != badId) {
    const CaloCellGeometry *thisCell = geometry.getGeometry(southeast);
    if(!thisCell) {
      LogError("PFClusterProducer")
	<<"warning detid "<<detid.rawId()
	<<" not found in geometry"<<endl;
    }
    else {
      const GlobalPoint& pos = thisCell->getPosition();
      double cornerposx = pos.x(); 
      double cornerposy = pos.y(); 
      double cornerposz = pos.z(); 
      cornerposx += rhposx; cornerposx /= 2.;
      cornerposy += rhposy; cornerposy /= 2.;
      cornerposz += rhposz; cornerposz /= 2.;
            
      rh->setSWCorner(cornerposx, cornerposy, cornerposz);
    }
  }
 
   
  if(southwest != badId) {
    const CaloCellGeometry *thisCell = geometry.getGeometry(southwest);
    if(!thisCell) {
      LogError("PFClusterProducer")
	<<"warning detid "<<detid.rawId()
	<<" not found in geometry"<<endl;
    }
    else {
      const GlobalPoint& pos = thisCell->getPosition();
      double cornerposx = pos.x(); 
      double cornerposy = pos.y(); 
      double cornerposz = pos.z(); 
      cornerposx += rhposx; cornerposx /= 2.;
      cornerposy += rhposy; cornerposy /= 2.;
      cornerposz += rhposz; cornerposz /= 2.;
            
      rh->setSECorner(cornerposx, cornerposy, cornerposz);
    }
  }
 
   
  if(northwest != badId) {
    const CaloCellGeometry *thisCell = geometry.getGeometry(northwest);
    if(!thisCell) {
      LogError("PFClusterProducer")
	<<"warning detid "<<detid.rawId()
	<<" not found in geometry"<<endl;
    }
    else {
      const GlobalPoint& pos = thisCell->getPosition();
      double cornerposx = pos.x(); 
      double cornerposy = pos.y(); 
      double cornerposz = pos.z(); 
      cornerposx += rhposx; cornerposx /= 2.;
      cornerposy += rhposy; cornerposy /= 2.;
      cornerposz += rhposz; cornerposz /= 2.;
            
      rh->setNECorner(cornerposx, cornerposy, cornerposz);
    }
  }
 
   


  // find and set neighbours

  reco::PFRecHit* rhnorth = 0;
  PFClusterAlgo::IDH i = rechits.find( north.rawId() );
  if(i != rechits.end() ) 
    rhnorth = i->second;
    
  reco::PFRecHit* rhnortheast = 0;
  i = rechits.find( northeast.rawId() );
  if(i != rechits.end() ) 
    rhnortheast = i->second;
    
  reco::PFRecHit* rhsouth = 0;
  i = rechits.find( south.rawId() );
  if(i != rechits.end() ) 
    rhsouth = i->second;
    
  reco::PFRecHit* rhsouthwest = 0;
  i = rechits.find( southwest.rawId() );
  if(i != rechits.end() ) 
    rhsouthwest = i->second;
  
  reco::PFRecHit* rheast = 0;
  i = rechits.find( east.rawId() );
  if(i != rechits.end() ) 
    rheast = i->second;
  
  reco::PFRecHit* rhsoutheast = 0;
  i = rechits.find( southeast.rawId() );
  if(i != rechits.end() ) 
    rhsoutheast = i->second;
  
  reco::PFRecHit* rhwest = 0;
  i = rechits.find( west.rawId() );
  if(i != rechits.end() ) 
    rhwest = i->second;
  
  reco::PFRecHit* rhnorthwest = 0;
  i = rechits.find( northwest.rawId() );
  if(i != rechits.end() ) 
    rhnorthwest = i->second;
  
  vector<reco::PFRecHit*> neighbours;
  neighbours.reserve(8);
  neighbours.push_back( rhnorth );
  neighbours.push_back( rhnorthwest );
  neighbours.push_back( rhwest );
  neighbours.push_back( rhsouthwest );
  neighbours.push_back( rhsouth );
  neighbours.push_back( rhsoutheast );
  neighbours.push_back( rheast );
  neighbours.push_back( rhnortheast );
  
  rh->setNeighbours( neighbours );
 
  
  if(westids.size()==2 || eastids.size()==2) 
    debug = true;
  
  if(debug) {
    
    cout<<"NWSE "
	<<northids.size()<<" "
	<<westids.size()<<" "
	<<southids.size()<<" "
	<<eastids.size()<<" "
	<<rhpos.Eta()<<" "<<rhpos.Phi()<<endl;
    
    cout<<"west neighbours:"<<endl;
    for(unsigned i=0; i<westids.size(); i++) {
      const CaloCellGeometry *thisCell = geometry.getGeometry( westids[i] );
      const GlobalPoint& npos = thisCell->getPosition();
      cout<<"\t"<<i<<": "
	  <<npos.eta()<<" "
	  <<npos.phi()<<endl;
    }
    cout<<"east neighbours:"<<endl;
    for(unsigned i=0; i<eastids.size(); i++) {
      const CaloCellGeometry *thisCell = geometry.getGeometry( eastids[i] );
      const GlobalPoint& npos = thisCell->getPosition();
    
      cout<<"\t"<<i<<": "
	  <<npos.eta()<<" "
	  <<npos.phi()<<endl;
    }
  }
}

DetId PFClusterProducer::getSouth(const DetId& id, 
				  const CaloSubdetectorTopology& topology) {

  DetId south;
  vector<DetId> sids = topology.south(id);
  if(sids.size() == 1)
    south = sids[0];
  
  return south;
} 

DetId PFClusterProducer::getNorth(const DetId& id, 
				  const CaloSubdetectorTopology& topology) {

  DetId north;
  vector<DetId> nids = topology.north(id);
  if(nids.size() == 1)
    north = nids[0];
  
  return north;
} 




//define this as a plug-in
DEFINE_FWK_MODULE(PFClusterProducer);
