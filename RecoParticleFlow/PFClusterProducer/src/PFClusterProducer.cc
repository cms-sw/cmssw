#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterProducer.h"

#include <memory>

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

    
  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);



  // parameters for ecal clustering
  
  double threshEcalBarrel = 
    iConfig.getParameter<double>("thresh_Ecal_Barrel");
  double threshSeedEcalBarrel = 
    iConfig.getParameter<double>("thresh_Seed_Ecal_Barrel");

  double threshEcalEndcap = 
    iConfig.getParameter<double>("thresh_Ecal_Endcap");
  double threshSeedEcalEndcap = 
    iConfig.getParameter<double>("thresh_Seed_Ecal_Endcap");


  int nNeighboursEcal = 
    iConfig.getParameter<int>("nNeighbours_Ecal");

  double posCalcP1Ecal = 
    iConfig.getParameter<double>("posCalcP1_Ecal");

  int posCalcNCrystalEcal = 
    iConfig.getParameter<int>("posCalcNCrystal_Ecal");
    
  double showerSigmaEcal = 
    iConfig.getParameter<double>("showerSigma_Ecal");
    

  clusterAlgoECAL_.setThreshBarrel( threshEcalBarrel );
  clusterAlgoECAL_.setThreshSeedBarrel( threshSeedEcalBarrel );
  
  clusterAlgoECAL_.setThreshEndcap( threshEcalEndcap );
  clusterAlgoECAL_.setThreshSeedEndcap( threshSeedEcalEndcap );

  clusterAlgoECAL_.setNNeighbours( nNeighboursEcal );
  clusterAlgoECAL_.setPosCalcP1( posCalcP1Ecal );
  clusterAlgoECAL_.setPosCalcNCrystal( posCalcNCrystalEcal );
  clusterAlgoECAL_.setShowerSigma( showerSigmaEcal );



  int dcormode = 
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



  // parameters for hcal clustering

  double threshHcalBarrel = 
    iConfig.getParameter<double>("thresh_Hcal_Barrel");
  double threshSeedHcalBarrel = 
    iConfig.getParameter<double>("thresh_Seed_Hcal_Barrel");

  double threshHcalEndcap = 
    iConfig.getParameter<double>("thresh_Hcal_Endcap");
  double threshSeedHcalEndcap = 
    iConfig.getParameter<double>("thresh_Seed_Hcal_Endcap");



  int nNeighboursHcal = 
    iConfig.getParameter<int>("nNeighbours_Hcal");

  double posCalcP1Hcal = 
    iConfig.getParameter<double>("posCalcP1_Hcal");

  int posCalcNCrystalHcal = 
    iConfig.getParameter<int>("posCalcNCrystal_Hcal");
    
  double showerSigmaHcal = 
    iConfig.getParameter<double>("showerSigma_Hcal");
  
  clusterAlgoHCAL_.setThreshBarrel( threshHcalBarrel );
  clusterAlgoHCAL_.setThreshSeedBarrel( threshSeedHcalBarrel );
  
  clusterAlgoHCAL_.setThreshEndcap( threshHcalEndcap );
  clusterAlgoHCAL_.setThreshSeedEndcap( threshSeedHcalEndcap );

  clusterAlgoHCAL_.setNNeighbours( nNeighboursHcal );
  clusterAlgoHCAL_.setPosCalcP1( posCalcP1Hcal );
  clusterAlgoHCAL_.setPosCalcNCrystal( posCalcNCrystalHcal );
  clusterAlgoHCAL_.setShowerSigma( showerSigmaHcal );
  

  // parameters for preshower clustering 

  double threshPS = 
    iConfig.getParameter<double>("thresh_PS");
  double threshSeedPS = 
    iConfig.getParameter<double>("thresh_Seed_PS");
  

  int nNeighboursPS = 
    iConfig.getParameter<int>("nNeighbours_PS");

  double posCalcP1PS = 
    iConfig.getParameter<double>("posCalcP1_PS");

  int posCalcNCrystalPS = 
    iConfig.getParameter<int>("posCalcNCrystal_PS");
    
  double showerSigmaPS = 
    iConfig.getParameter<double>("showerSigma_PS");
  
  
  clusterAlgoPS_.setThreshBarrel( threshPS );
  clusterAlgoPS_.setThreshSeedBarrel( threshSeedPS );
  
  clusterAlgoPS_.setThreshEndcap( threshPS );
  clusterAlgoPS_.setThreshSeedEndcap( threshSeedPS );

  clusterAlgoPS_.setNNeighbours( nNeighboursPS );
  clusterAlgoPS_.setPosCalcP1( posCalcP1PS );
  clusterAlgoPS_.setPosCalcNCrystal( posCalcNCrystalPS );
  clusterAlgoPS_.setShowerSigma( showerSigmaPS );


  // access to the collections of rechits from the various detectors:

  
  inputTagEcalRecHitsEB_ = 
    iConfig.getParameter<InputTag>("ecalRecHitsEB");

  inputTagEcalRecHitsEE_ = 
    iConfig.getParameter<InputTag>("ecalRecHitsEE");
  
  inputTagEcalRecHitsES_ = 
    iConfig.getParameter<InputTag>("ecalRecHitsES");
  

  inputTagHcalRecHitsHBHE_ =
    iConfig.getParameter<InputTag>("hcalRecHitsHBHE");
    
 
  inputTagCaloTowers_ = 
    iConfig.getParameter<InputTag>("caloTowers");
   
  
  neighbourmapcalculated_ = false;
    
  // produce PFRecHits yes/no
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
  

  if( processEcal_ ) {


    vector<reco::PFRecHit> *prechits = new vector<reco::PFRecHit>;
    vector<reco::PFRecHit>& rechits = *prechits;

    // create PFRecHits and put them in rechits
    createEcalRecHits( rechits, iEvent, iSetup);

    edm::OrphanHandle< reco::PFRecHitCollection > rechitsHandle;

    // if requested, write PFRecHits in the event
    if ( produceRecHits_ ) {
      auto_ptr< vector<reco::PFRecHit> > recHits( prechits ); 
      rechitsHandle = iEvent.put( recHits, "ECAL" );
    } else {
      rechitsHandle = edm::OrphanHandle< reco::PFRecHitCollection >( &rechits, edm::ProductID(10001) );
    }

    if(clusteringEcal_) {
      
      // do clustering
      clusterAlgoECAL_.doClustering( rechitsHandle );
      
      if( verbose_ ) {
	LogInfo("PFClusterProducer")
	  <<" ECAL clusters --------------------------------- "<<endl
	  <<clusterAlgoECAL_<<endl;
      }    
    
      // get clusters out of the clustering algorithm 
      // and put them in the event. There is no copy.
      auto_ptr< vector<reco::PFCluster> > 
	outClustersECAL( clusterAlgoECAL_.clusters() ); 
      iEvent.put( outClustersECAL, "ECAL");
    }    
  }


  if( processHcal_ ) {
    
    vector<reco::PFRecHit> *prechits = new vector<reco::PFRecHit>;
    vector<reco::PFRecHit>& rechits = *prechits;

    createHcalRecHits(rechits, iEvent, iSetup);

    edm::OrphanHandle< reco::PFRecHitCollection > rechitsHandle;

    // if requested, write PFRecHits in the event
    if ( produceRecHits_ ) {
      auto_ptr< vector<reco::PFRecHit> > recHits( prechits ); 
      rechitsHandle = iEvent.put( recHits, "HCAL" );
    } else {
      rechitsHandle = edm::OrphanHandle< reco::PFRecHitCollection >( &rechits, edm::ProductID(10002) );
    }
    
    if(clusteringHcal_) {
      
      // do clustering
      clusterAlgoHCAL_.doClustering( rechitsHandle );
      
      if(verbose_) {
	LogInfo("PFClusterProducer")
	  <<" HCAL clusters --------------------------------- "<<endl
	  <<clusterAlgoHCAL_<<endl;
      }      

      // get clusters out of the clustering algorithm 
      // and put them in the event. There is no copy.
      auto_ptr< vector<reco::PFCluster> > 
	outClustersHCAL( clusterAlgoHCAL_.clusters() ); 
      iEvent.put( outClustersHCAL, "HCAL");
    }
  }


  if( processPS_ ) {

    vector<reco::PFRecHit> *prechits = new vector<reco::PFRecHit>;
    vector<reco::PFRecHit>& rechits = *prechits;

    createPSRecHits(rechits, iEvent, iSetup);

    edm::OrphanHandle< reco::PFRecHitCollection > rechitsHandle;

    // if requested, write PFRecHits in the event
    if ( produceRecHits_ ) {
      auto_ptr< vector<reco::PFRecHit> > recHits( prechits ); 
      rechitsHandle = iEvent.put( recHits, "PS" );
    } else {
      rechitsHandle = edm::OrphanHandle< reco::PFRecHitCollection >( &rechits, edm::ProductID(10003) );
    }

    if(clusteringPS_) {
  
      clusterAlgoPS_.doClustering( rechitsHandle );

      if(verbose_) {
	LogInfo("PFClusterProducer")
	  <<" Preshower clusters --------------------------------- "<<endl
	  <<clusterAlgoPS_<<endl;   
      }

      // get clusters out of the clustering algorithm 
      // and put them in the event. There is no copy.
      auto_ptr< vector<reco::PFCluster> > 
	outClustersPS( clusterAlgoPS_.clusters() ); 
      iEvent.put( outClustersPS, "PS");
    }    
  }
}


void PFClusterProducer::createEcalRecHits(vector<reco::PFRecHit>& rechits,
					  edm::Event& iEvent, 
					  const edm::EventSetup& iSetup ) {



  // this map is necessary to find the rechit neighbours efficiently
  //C but I should think about using Florian's hashed index to do this.
  //C in which case the map might not be necessary anymore
  // 
  // the key of this map is detId. 
  // the value is the index in the rechits vector
  map<unsigned, unsigned > idSortedRecHits;
  typedef map<unsigned, unsigned >::iterator IDH;

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

    
  if(!neighbourmapcalculated_)
    ecalNeighbArray( *ecalBarrelGeometry,
		     ecalBarrelTopology,
		     *ecalEndcapGeometry,
		     ecalEndcapTopology );

         
  // get the ecalBarrel rechits

  edm::Handle<EcalRecHitCollection> rhcHandle;


  bool found = iEvent.getByLabel(inputTagEcalRecHitsEB_, 
				 rhcHandle);
  
  if(!found) {

    ostringstream err;
    err<<"could not find rechits "<<inputTagEcalRecHitsEB_;
    LogError("PFClusterProducer")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }
  else {
    assert( rhcHandle.isValid() );
    
    // process ecal ecalBarrel rechits
    for(unsigned i=0; i<rhcHandle->size(); i++) {
      
      const EcalRecHit& erh = (*rhcHandle)[i];
      const DetId& detid = erh.detid();
      double energy = erh.energy();
      EcalSubdetector esd=(EcalSubdetector)detid.subdetId();
      if (esd != 1) continue;
      if(energy < clusterAlgoECAL_.threshBarrel() ) continue;
          
      
      reco::PFRecHit *pfrh = createEcalRecHit(detid, energy,  
					      PFLayer::ECAL_BARREL,
					      ecalBarrelGeometry);
      
      if( !pfrh ) continue; // problem with this rechit. skip it
      
      rechits.push_back( *pfrh );
      delete pfrh;
      idSortedRecHits.insert( make_pair(detid.rawId(), rechits.size()-1 ) ); 
    }      
  }



  //C proceed as for the barrel
  // process ecal endcap rechits

  found = iEvent.getByLabel(inputTagEcalRecHitsEE_,
			    rhcHandle);
  
  if(!found) {
    ostringstream err;
    err<<"could not find rechits "<<inputTagEcalRecHitsEE_;
    LogError("PFClusterProducer")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }
  else {
    assert( rhcHandle.isValid() );
    
    for(unsigned i=0; i<rhcHandle->size(); i++) {
      
      const EcalRecHit& erh = (*rhcHandle)[i];
      const DetId& detid = erh.detid();
      double energy = erh.energy();
      EcalSubdetector esd=(EcalSubdetector)detid.subdetId();
      if (esd != 2) continue;
      if(energy < clusterAlgoECAL_.threshEndcap() ) continue;

      
      reco::PFRecHit *pfrh = createEcalRecHit(detid, energy,
					      PFLayer::ECAL_ENDCAP,
					      ecalEndcapGeometry);
      if( !pfrh ) continue; // problem with this rechit. skip it

      rechits.push_back( *pfrh );
      delete pfrh;
      idSortedRecHits.insert( make_pair(detid.rawId(), rechits.size()-1 ) ); 
    }
  }


  // do navigation
  for(unsigned i=0; i<rechits.size(); i++ ) {
    
//     findRecHitNeighbours( rechits[i], idSortedRecHits, 
// 			  ecalBarrelTopology, 
// 			  *ecalBarrelGeometry, 
// 			  ecalEndcapTopology,
// 			  *ecalEndcapGeometry);
    findRecHitNeighboursECAL( rechits[i], idSortedRecHits ); 
			      
  }
} 

  

void PFClusterProducer::createHcalRecHits(vector<reco::PFRecHit>& rechits,
					  edm::Event& iEvent, 
					  const edm::EventSetup& iSetup ) {

  
  // this map is necessary to find the rechit neighbours efficiently
  //C but I should think about using Florian's hashed index to do this.
  //C in which case the map might not be necessary anymore
  //C however the hashed index does not seem to be implemented for HCAL
  // 
  // the key of this map is detId. 
  // the value is the index in the rechits vector
  map<unsigned,  unsigned > idSortedRecHits;
  typedef map<unsigned, unsigned >::iterator IDH;  


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

    // get calotowers
    bool found = iEvent.getByLabel(inputTagCaloTowers_,
				   caloTowers);

    if(!found) {
      ostringstream err;
      err<<"could not find rechits "<<inputTagCaloTowers_;
      LogError("PFClusterProducer")<<err.str()<<endl;
    
      throw cms::Exception( "MissingProduct", err.str());
    }
    else {
      assert( caloTowers.isValid() );
      
      // create rechits
      typedef CaloTowerCollection::const_iterator ICT;
      
      for(ICT ict=caloTowers->begin(); ict!=caloTowers->end();ict++) {
	  
	const CaloTower& ct = (*ict);
	  
	//C	
	if(!caloTowerGeometry) 
	  caloTowerGeometry = geoHandle->getSubdetectorGeometry(ct.id());
	  
	// get the hadronic energy.
	double energy = ct.hadEnergy();
	if( energy < 1e-9 ) continue;  
	  
	  
	  
	// the layer will be taken from the first constituent. 
	// all thresholds for ECAL must be set to very high values !!!
	assert( ct.constituentsSize() );	  
	const HcalDetId& detid = ct.constituent(0);
	  
	reco::PFRecHit* pfrh = 0;
	  
	switch( detid.subdet() ) {
	case HcalBarrel:
	  if(energy > clusterAlgoHCAL_.threshBarrel() ) {
	    pfrh = createHcalRecHit( detid, 
				     energy, 
				     PFLayer::HCAL_BARREL1, 
				     hcalBarrelGeometry,
				     ct.id().rawId() );
	  }
	  break;
	case HcalEndcap:
	  if(energy > clusterAlgoHCAL_.threshEndcap() ) {
	    pfrh = createHcalRecHit( detid, 
				     energy, 
				     PFLayer::HCAL_ENDCAP, 
				     hcalEndcapGeometry,
				     ct.id().rawId() );
	  }
	  break;
	default:
	  LogError("PFClusterProducer")
	    <<"CaloTower constituent: unknown layer : "
	    <<detid.subdet()<<endl;
	} 
	  
	if(pfrh) { 
	  rechits.push_back( *pfrh );
	  delete pfrh;
	  idSortedRecHits.insert( make_pair(ct.id().rawId(), 
					    rechits.size()-1 ) ); 
	}
      }
	
	
      // do navigation 
      for(unsigned i=0; i<rechits.size(); i++ ) {
	  
	findRecHitNeighboursCT( rechits[i], 
				idSortedRecHits, 
				caloTowerTopology);
	  
      }
    }   
  }
  else { // clustering is not done on CaloTowers but on HCAL rechits.
       

    // get the hcal topology
    HcalTopology hcalTopology;
    
    // HCAL rechits 
    //    vector<edm::Handle<HBHERecHitCollection> > hcalHandles;  
    edm::Handle<HBHERecHitCollection>  hcalHandle;  

    
    bool found = iEvent.getByLabel(inputTagHcalRecHitsHBHE_, 
				   hcalHandle );

    if(!found) {
      ostringstream err;
      err<<"could not find rechits "<<inputTagHcalRecHitsHBHE_;
      LogError("PFClusterProducer")<<err.str()<<endl;
    
      throw cms::Exception( "MissingProduct", err.str());
    }
    else {
      assert( hcalHandle.isValid() );
      
      const edm::Handle<HBHERecHitCollection>& handle = hcalHandle;
      for(unsigned irechit=0; irechit<handle->size(); irechit++) {
	const HBHERecHit& hit = (*handle)[irechit];
	
	double energy = hit.energy();
	
	reco::PFRecHit* pfrh = 0;
	

	const HcalDetId& detid = hit.detid();
	switch( detid.subdet() ) {
	case HcalBarrel:
	  if(energy > clusterAlgoHCAL_.threshBarrel() ){
	    pfrh = createHcalRecHit( detid, 
				     energy, 
				     PFLayer::HCAL_BARREL1, 
				     hcalBarrelGeometry );
	  }
	  break;
	case HcalEndcap:
	  if(energy > clusterAlgoHCAL_.threshEndcap() ){
	    pfrh = createHcalRecHit( detid, 
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

	if(pfrh) { 
	  rechits.push_back( *pfrh );
	  delete pfrh;
	  idSortedRecHits.insert( make_pair(detid.rawId(), 
					    rechits.size()-1 ) ); 
	}
      }
      
      
      // do navigation:
      for(unsigned i=0; i<rechits.size(); i++ ) {
	
	findRecHitNeighbours( rechits[i], idSortedRecHits, 
			      hcalTopology, 
			      *hcalBarrelGeometry, 
			      hcalTopology,
			      *hcalEndcapGeometry);
      } // loop for navigation
    }  // endif hcal rechits were found
  } // endif clustering on rechits in hcal
}



void PFClusterProducer::createPSRecHits(vector<reco::PFRecHit>& rechits,
					edm::Event& iEvent, 
					const edm::EventSetup& iSetup) {

  map<unsigned, unsigned > idSortedRecHits;
  typedef map<unsigned, unsigned >::iterator IDH;


  // get the ps geometry
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<IdealGeometryRecord>().get(geoHandle);
    
  const CaloSubdetectorGeometry *psGeometry = 
    geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
    
  // get the ps topology
  EcalPreshowerTopology psTopology(geoHandle);

  // process rechits
  Handle< EcalRecHitCollection >   pRecHits;



  bool found = iEvent.getByLabel(inputTagEcalRecHitsES_,
				 pRecHits);

  if(!found) {
    ostringstream err;
    err<<"could not find rechits "<<inputTagEcalRecHitsES_;
    LogError("PFClusterProducer")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }
  else {
    assert( pRecHits.isValid() );

    const EcalRecHitCollection& psrechits = *( pRecHits.product() );
    typedef EcalRecHitCollection::const_iterator IT;
 
    for(IT i=psrechits.begin(); i!=psrechits.end(); i++) {
      const EcalRecHit& hit = *i;
      
      double energy = hit.energy();
      if( energy < clusterAlgoPS_.threshEndcap() ) continue; 
            
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
 
      reco::PFRecHit *pfrh
	= new reco::PFRecHit( detid.rawId(), layer, energy, 
			      position.x(), position.y(), position.z(), 
			      0,0,0 );
      
      const CaloCellGeometry::CornersVec& corners = thisCell->getCorners();
      assert( corners.size() == 8 );
      
      pfrh->setNECorner( corners[0].x(), corners[0].y(),  corners[0].z() );
      pfrh->setSECorner( corners[1].x(), corners[1].y(),  corners[1].z() );
      pfrh->setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z() );
      pfrh->setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z() );

      
      // if( !pfrh ) continue; // problem with this rechit. skip it

      rechits.push_back( *pfrh );
      delete pfrh;
      idSortedRecHits.insert( make_pair(detid.rawId(), rechits.size()-1 ) );   
    }
  }

  // do navigation
  for(unsigned i=0; i<rechits.size(); i++ ) {
    
    findRecHitNeighbours( rechits[i], idSortedRecHits, 
			  psTopology, 
			  *psGeometry, 
			  psTopology,
			  *psGeometry);
  }
}



reco::PFRecHit* 
PFClusterProducer::createEcalRecHit( const DetId& detid,
				     double energy,
				     int layer,
				     const CaloSubdetectorGeometry* geom ) {

  math::XYZVector position;
  math::XYZVector axis;

  const CaloCellGeometry *thisCell 
    = geom->getGeometry(detid);
  
  // find rechit geometry
  if(!thisCell) {
    LogError("PFClusterProducer")
      <<"warning detid "<<detid.rawId()
      <<" not found in geometry"<<endl;
    return 0;
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
  }
  else return 0;

//   if( !geomfound ) {
//     LogError("PFClusterProducer")<<"cannor find geometry for detid "
// 				 <<detid.rawId()<<" in layer "<<layer<<endl;
//     return 0; // geometry not found, skip rechit
//   }
  
  
  reco::PFRecHit *rh 
    = new reco::PFRecHit( detid.rawId(), layer, 
			  energy, 
			  position.x(), position.y(), position.z(), 
			  axis.x(), axis.y(), axis.z() ); 


  const CaloCellGeometry::CornersVec& corners = thisCell->getCorners();
  assert( corners.size() == 8 );

  rh->setNECorner( corners[0].x(), corners[0].y(),  corners[0].z() );
  rh->setSECorner( corners[1].x(), corners[1].y(),  corners[1].z() );
  rh->setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z() );
  rh->setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z() );

  return rh;
}



reco::PFRecHit* 
PFClusterProducer::createHcalRecHit( const DetId& detid,
				     double energy,
				     int layer,
				     const CaloSubdetectorGeometry* geom,
				     unsigned newDetId ) {
  
  const CaloCellGeometry *thisCell = geom->getGeometry(detid);
  if(!thisCell) {
    edm::LogError("PFClusterProducer")
      <<"warning detid "<<detid.rawId()<<" not found in layer "
      <<layer<<endl;
    return 0;
  }
  
  const GlobalPoint& position = thisCell->getPosition();
  
  unsigned id = detid;
  if(newDetId) id = newDetId;
  reco::PFRecHit *rh = 
    new reco::PFRecHit( id,  layer, energy, 
			position.x(), position.y(), position.z(), 
			0,0,0 );
 
  // set the corners
  const CaloCellGeometry::CornersVec& corners = thisCell->getCorners();

  assert( corners.size() == 8 );

  rh->setNECorner( corners[0].x(), corners[0].y(),  corners[0].z() );
  rh->setSECorner( corners[1].x(), corners[1].y(),  corners[1].z() );
  rh->setSWCorner( corners[2].x(), corners[2].y(),  corners[2].z() );
  rh->setNWCorner( corners[3].x(), corners[3].y(),  corners[3].z() );
 
  return rh;
}



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
PFClusterProducer::findRecHitNeighboursECAL
( reco::PFRecHit& rh, 
  const map<unsigned,unsigned >& sortedHits ) {
  
  DetId center( rh.detId() );


  DetId north = move( center, NORTH );
  DetId northeast = move( center, NORTHEAST );
  DetId northwest = move( center, NORTHWEST ); 
  DetId south = move( center, SOUTH );  
  DetId southeast = move( center, SOUTHEAST );  
  DetId southwest = move( center, SOUTHWEST );  
  DetId east  = move( center, EAST );  
  DetId west  = move( center, WEST );  
    
  PFClusterAlgo::IDH i = sortedHits.find( north.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
  
  i = sortedHits.find( northeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
  
  i = sortedHits.find( south.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( east.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( west.rawId() );
  if(i != sortedHits.end() ) 
     rh.add4Neighbour( i->second );
   
  i = sortedHits.find( northwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
}



void 
PFClusterProducer::findRecHitNeighbours
( reco::PFRecHit& rh, 
  const map<unsigned,unsigned >& sortedHits, 
  const CaloSubdetectorTopology& barrelTopology, 
  const CaloSubdetectorGeometry& barrelGeometry, 
  const CaloSubdetectorTopology& endcapTopology, 
  const CaloSubdetectorGeometry& endcapGeometry ) {
  
  DetId detid( rh.detId() );

  const CaloSubdetectorTopology* topology = 0;
  const CaloSubdetectorGeometry* geometry = 0;
  const CaloSubdetectorGeometry* othergeometry = 0;
  
  switch( rh.layer() ) {
  case PFLayer::ECAL_ENDCAP: 
    topology = &endcapTopology;
    geometry = &endcapGeometry;
    break;
  case PFLayer::ECAL_BARREL: 
    topology = &barrelTopology;
    geometry = &barrelGeometry;
    break;
  case PFLayer::HCAL_ENDCAP:
    topology = &endcapTopology;
    geometry = &endcapGeometry;
    othergeometry = &barrelGeometry;
    break;
  case PFLayer::HCAL_BARREL1:
    topology = &barrelTopology;
    geometry = &barrelGeometry;
    othergeometry = &endcapGeometry;
    break;
  case PFLayer::PS1:
  case PFLayer::PS2:
    topology = &barrelTopology;
    geometry = &barrelGeometry;
    othergeometry = &endcapGeometry;
    break;
  default:
    assert(0);
  }
  
  assert( topology && geometry );

  CaloNavigator<DetId> navigator(detid, topology);

  DetId north = navigator.north();  
  
  DetId northeast(0);
  if( north != DetId(0) ) {
    northeast = navigator.east();  
  }
  navigator.home();


  DetId south = navigator.south();

  

  DetId southwest(0); 
  if( south != DetId(0) ) {
    southwest = navigator.west();
  }
  navigator.home();


  DetId east = navigator.east();
  DetId southeast;
  if( east != DetId(0) ) {
    southeast = navigator.south(); 
  }
  navigator.home();
  DetId west = navigator.west();
  DetId northwest;
  if( west != DetId(0) ) {   
    northwest = navigator.north();  
  }
  navigator.home();
    
  PFClusterAlgo::IDH i = sortedHits.find( north.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
  
  i = sortedHits.find( northeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
  
  i = sortedHits.find( south.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( east.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( west.rawId() );
  if(i != sortedHits.end() ) 
     rh.add4Neighbour( i->second );
   
  i = sortedHits.find( northwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    

}


void 
PFClusterProducer::findRecHitNeighboursCT
( reco::PFRecHit& rh, 
  const map<unsigned, unsigned >& sortedHits, 
  const CaloSubdetectorTopology& topology ) {

  CaloTowerDetId ctDetId( rh.detId() );
  

  vector<DetId> northids = topology.north(ctDetId);
  vector<DetId> westids = topology.west(ctDetId);
  vector<DetId> southids = topology.south(ctDetId);
  vector<DetId> eastids = topology.east(ctDetId);


  CaloTowerDetId badId;

  // all the following detids will be CaloTowerDetId
  CaloTowerDetId north;
  CaloTowerDetId northwest;
  CaloTowerDetId west;
  CaloTowerDetId southwest;
  CaloTowerDetId south;
  CaloTowerDetId southeast;
  CaloTowerDetId east;
  CaloTowerDetId northeast;
  
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




  // find and set neighbours
    
  PFClusterAlgo::IDH i = sortedHits.find( north.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
  
  i = sortedHits.find( northeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
  
  i = sortedHits.find( south.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( east.rawId() );
  if(i != sortedHits.end() ) 
    rh.add4Neighbour( i->second );
    
  i = sortedHits.find( southeast.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
    
  i = sortedHits.find( west.rawId() );
  if(i != sortedHits.end() ) 
     rh.add4Neighbour( i->second );
   
  i = sortedHits.find( northwest.rawId() );
  if(i != sortedHits.end() ) 
    rh.add8Neighbour( i->second );
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



// Build the array of (max)8 neighbors
void 
PFClusterProducer::ecalNeighbArray(const CaloSubdetectorGeometry& barrelGeom,
				   const CaloSubdetectorTopology& barrelTopo,
				   const CaloSubdetectorGeometry& endcapGeom,
				   const CaloSubdetectorTopology& endcapTopo ){
  

  static const CaloDirection orderedDir[8]={SOUTHWEST,
					    SOUTH,
					    SOUTHEAST,
					    WEST,
					    EAST,
					    NORTHWEST,
					    NORTH,
                                            NORTHEAST};

  const unsigned nbarrel = 62000;
  // Barrel first. The hashed index runs from 0 to 61199
  neighboursEB_.resize(nbarrel);
  
  //std::cout << " Building the array of neighbours (barrel) " ;

  std::vector<DetId> vec(barrelGeom.getValidDetIds(DetId::Ecal,
						   EcalBarrel));
  unsigned size=vec.size();    
  for(unsigned ic=0; ic<size; ++ic) 
    {
      // We get the 9 cells in a square. 
      std::vector<DetId> neighbours(barrelTopo.getWindow(vec[ic],3,3));
      //      std::cout << " Cell " << EBDetId(vec[ic]) << std::endl;
      unsigned nneighbours=neighbours.size();

      unsigned hashedindex=EBDetId(vec[ic]).hashedIndex();
      if(hashedindex>=nbarrel)
        {
          LogDebug("CaloGeometryTools")  << " Array overflow " << std::endl;
        }


      // If there are 9 cells, it is easy, and this order is know:
//      6  7  8
//      3  4  5 
//      0  1  2   (0 = SOUTHWEST)

      if(nneighbours==9)
        {
          neighboursEB_[hashedindex].reserve(8);
          for(unsigned in=0;in<nneighbours;++in)
            {
              // remove the centre
              if(neighbours[in]!=vec[ic]) 
                {
                  neighboursEB_[hashedindex].push_back(neighbours[in]);
                  //          std::cout << " Neighbour " << in << " " << EBDetId(neighbours[in]) << std::endl;
                }
            }
        }
      else
        {
          DetId central(vec[ic]);
          neighboursEB_[hashedindex].resize(8,DetId(0));
          for(unsigned idir=0;idir<8;++idir)
            {
              DetId testid=central;
              bool status=stdmove(testid,orderedDir[idir],
				  barrelTopo, endcapTopo);
              if(status) neighboursEB_[hashedindex][idir]=testid;
            }

        }
    }

  // Moved to the endcap

  //  std::cout << " done " << size << std::endl;
  //  std::cout << " Building the array of neighbours (endcap) " ;

  vec.clear();
  vec=endcapGeom.getValidDetIds(DetId::Ecal,EcalEndcap);
  size=vec.size();    
  // There are some holes in the hashedIndex for the EE. Hence the array is bigger than the number
  // of crystals
  const unsigned nendcap=19960;

  neighboursEE_.resize(nendcap);
  for(unsigned ic=0; ic<size; ++ic) 
    {
      // We get the 9 cells in a square. 
      std::vector<DetId> neighbours(endcapTopo.getWindow(vec[ic],3,3));
      unsigned nneighbours=neighbours.size();
      // remove the centre
      unsigned hashedindex=EEDetId(vec[ic]).hashedIndex();
      
      if(hashedindex>=nendcap)
        {
          LogDebug("CaloGeometryTools")  << " Array overflow " << std::endl;
        }

      if(nneighbours==9)
        {
          neighboursEE_[hashedindex].reserve(8);
          for(unsigned in=0;in<nneighbours;++in)
            {     
              // remove the centre
              if(neighbours[in]!=vec[ic]) 
                {
                  neighboursEE_[hashedindex].push_back(neighbours[in]);
                }
            }
        }
      else
        {
          DetId central(vec[ic]);
          neighboursEE_[hashedindex].resize(8,DetId(0));
          for(unsigned idir=0;idir<8;++idir)
            {
              DetId testid=central;
              bool status=stdmove(testid,orderedDir[idir],
				  barrelTopo, endcapTopo );
              if(status) neighboursEE_[hashedindex][idir]=testid;
            }

        }
    }
  //  std::cout << " done " << size <<std::endl;
  neighbourmapcalculated_ = true;
}



bool 
PFClusterProducer::stdsimplemove(DetId& cell, 
				  const CaloDirection& dir,
				  const CaloSubdetectorTopology& barrelTopo,
				  const CaloSubdetectorTopology& endcapTopo ) 
  const {

  std::vector<DetId> neighbours;
  if(cell.subdetId()==EcalBarrel)
    neighbours = barrelTopo.getNeighbours(cell,dir);
  else if(cell.subdetId()==EcalEndcap)
    neighbours= endcapTopo.getNeighbours(cell,dir);
  
  if(neighbours.size()>0 && !neighbours[0].null())
    {
      cell = neighbours[0];
      return true;
    }
  else 
    {
      cell = DetId(0);
      return false;
    }
}



bool 
PFClusterProducer::stdmove(DetId& cell, 
			   const CaloDirection& dir,
			   const CaloSubdetectorTopology& barrelTopo,
			   const CaloSubdetectorTopology& endcapTopo ) 
  
  const {


  bool result; 

  if(dir==NORTH) {
    result = stdsimplemove(cell,NORTH, barrelTopo, endcapTopo );
    return result;
  }
  else if(dir==SOUTH) {
    result = stdsimplemove(cell,SOUTH, barrelTopo, endcapTopo );
    return result;
  }
  else if(dir==EAST) {
    result = stdsimplemove(cell,EAST, barrelTopo, endcapTopo );
    return result;
  }
  else if(dir==WEST) {
    result = stdsimplemove(cell,WEST, barrelTopo, endcapTopo );
    return result;
  }


  // One has to try both paths
  else if(dir==NORTHEAST)
    {
      result = stdsimplemove(cell,NORTH, barrelTopo, endcapTopo );
      if(result)
        return stdsimplemove(cell,EAST, barrelTopo, endcapTopo );
      else
        {
          result = stdsimplemove(cell,EAST, barrelTopo, endcapTopo );
          if(result)
            return stdsimplemove(cell,NORTH, barrelTopo, endcapTopo );
          else
            return false; 
        }
    }
  else if(dir==NORTHWEST)
    {
      result = stdsimplemove(cell,NORTH, barrelTopo, endcapTopo );
      if(result)
        return stdsimplemove(cell,WEST, barrelTopo, endcapTopo );
      else
        {
          result = stdsimplemove(cell,WEST, barrelTopo, endcapTopo );
          if(result)
            return stdsimplemove(cell,NORTH, barrelTopo, endcapTopo );
          else
            return false; 
        }
    }
  else if(dir == SOUTHEAST)
    {
      result = stdsimplemove(cell,SOUTH, barrelTopo, endcapTopo );
      if(result)
        return stdsimplemove(cell,EAST, barrelTopo, endcapTopo );
      else
        {
          result = stdsimplemove(cell,EAST, barrelTopo, endcapTopo );
          if(result)
            return stdsimplemove(cell,SOUTH, barrelTopo, endcapTopo );
          else
            return false; 
        }
    }
  else if(dir == SOUTHWEST)
    {
      result = stdsimplemove(cell,SOUTH, barrelTopo, endcapTopo );
      if(result)
        return stdsimplemove(cell,WEST, barrelTopo, endcapTopo );
      else
        {
          result = stdsimplemove(cell,SOUTH, barrelTopo, endcapTopo );
          if(result)
            return stdsimplemove(cell,WEST, barrelTopo, endcapTopo );
          else
            return false; 
        }
    }
  cell = DetId(0);
  return false;
}





DetId PFClusterProducer::move(DetId cell, 
			      const CaloDirection&dir ) const
{  
  DetId originalcell = cell; 
  if(dir==NONE || cell==DetId(0)) return false;

  // Conversion CaloDirection and index in the table
  // CaloDirection :NONE,SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST, NORTHEAST,NORTHWEST,NORTH
  // Table : SOUTHWEST,SOUTH,SOUTHEAST,WEST,EAST,NORTHWEST,NORTH, NORTHEAST
  static const int calodirections[9]={-1,1,2,0,4,3,7,5,6};
    
  assert(neighbourmapcalculated_);

  DetId result = (originalcell.subdetId()==EcalBarrel) ? 
    neighboursEB_[EBDetId(originalcell).hashedIndex()][calodirections[dir]]:
    neighboursEE_[EEDetId(originalcell).hashedIndex()][calodirections[dir]];
  return result; 
}





//define this as a plug-in
DEFINE_FWK_MODULE(PFClusterProducer);

