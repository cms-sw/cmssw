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

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//

using namespace std;

PFClusterProducer::PFClusterProducer(const edm::ParameterSet& iConfig)
{
  //register your products
  // produces<reco::PFClusterCollection>();
  produces< reco::PFRecHitCollection >();
  
  /* Examples
     produces<ExampleData2>();

     //if do put with a label
     produces<ExampleData2>("label");
  */
  //now do what ever other initialization is needed

  processEcal_ = 
    iConfig.getUntrackedParameter<bool>("process_Ecal",true);
  processHcal_ = 
    iConfig.getUntrackedParameter<bool>("process_Hcal",true);
  processPS_ = 
    iConfig.getUntrackedParameter<bool>("process_PS",true);

  // parameters for ecal clustering
  
  threshEcalBarrel_ = 
    iConfig.getUntrackedParameter<double>("thresh_Ecal_Barrel",0.5);
  threshSeedEcalBarrel_ = 
    iConfig.getUntrackedParameter<double>("thresh_Seed_Ecal_Barrel",0.3);

  threshEcalEndcap_ = 
    iConfig.getUntrackedParameter<double>("thresh_Ecal_Endcap",0.3);
  threshSeedEcalEndcap_ = 
    iConfig.getUntrackedParameter<double>("thresh_Seed_Ecal_Endcap",0.8);

  // parameters for preshower clustering 

  threshPS_ = 
    iConfig.getUntrackedParameter<double>("thresh_PS",0.0007);
  threshSeedPS_ = 
    iConfig.getUntrackedParameter<double>("thresh_Seed_PS",0.01);
  

  // parameters for hcal clustering

  threshHcalBarrel_ = 
    iConfig.getUntrackedParameter<double>("thresh_Hcal_Barrel",1);
  threshSeedHcalBarrel_ = 
    iConfig.getUntrackedParameter<double>("thresh_Seed_Hcal_Barrel",1.4);

  threshHcalEndcap_ = 
    iConfig.getUntrackedParameter<double>("thresh_Hcal_Endcap",1);
  threshSeedHcalEndcap_ = 
    iConfig.getUntrackedParameter<double>("thresh_Seed_Hcal_Endcap",1.4);

  

  int    dcormode = 
    iConfig.getUntrackedParameter<int>("depthCor_Mode",-1);
  
  double dcora = 
    iConfig.getUntrackedParameter<double>("depthCor_A",-1);
  double dcorb = 
    iConfig.getUntrackedParameter<double>("depthCor_B",-1);
  double dcorap = 
    iConfig.getUntrackedParameter<double>("depthCor_A_preshower",-1);
  double dcorbp = 
    iConfig.getUntrackedParameter<double>("depthCor_B_preshower",-1);

  if( dcormode > -0.5 && 
      dcora > -0.5 && 
      dcorb > -0.5 && 
      dcorap > -0.5 && 
      dcorbp > -0.5 )
    reco::PFCluster::SetDepthCorParameters( dcormode, dcora, dcorb, dcorap, dcorbp);
}


PFClusterProducer::~PFClusterProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
  
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void PFClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  cout<<"processing event "<<iEvent.id().event()
      <<" in run "<<iEvent.id().run()<<endl;
  
  auto_ptr< vector<reco::PFRecHit> > result(new vector<reco::PFRecHit> ); 


  

  if( processEcal_ ) {

    map<unsigned,  reco::PFRecHit* > ecalrechits;

    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<IdealGeometryRecord>().get(geoHandle);
    
    // get the ecalBarrel geometry
    const CaloSubdetectorGeometry *ecalBarrelGeometry = 
      geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    
    // get the ecalBarrel topology
    EcalBarrelTopology ecalBarrelTopology(geoHandle);
    
    // get the endcap geometry
    const CaloSubdetectorGeometry *endcapGeometry = 
      geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

    // get the endcap topology
    EcalEndcapTopology endcapTopology(geoHandle);

         
    // get the ecal ecalBarrel rechits
    edm::Handle<EcalRecHitCollection> rhcHandle;
    try {
      iEvent.getByLabel("ecalrechit", "EcalRecHitsEB", rhcHandle);
      if (!(rhcHandle.isValid())) {
	cout<<"could not get a handle on EcalRecHitsEB!"<<endl;
	return;
      }
    }
    catch ( cms::Exception& ex ) {
      edm::LogError("PFClusterProducerError")
	<<"Error! can't get the rechits "<<ex.what()<<endl;
      return;
    }

    // process ecal ecalBarrel rechits
    // cout<<"\trechits : "<<endl;
    for(unsigned i=0; i<rhcHandle->size(); i++) {
     
      double energy = (*rhcHandle)[i].energy();
      
      if(energy < threshEcalBarrel_ ) continue;
      
      const DetId& detid = (*rhcHandle)[i].detid();
      const CaloCellGeometry *thisCell = ecalBarrelGeometry->getGeometry(detid);
      
      if(!thisCell) {
	cerr<<"warning detid "<<detid.rawId()<<" not found in endcap geometry"<<endl;
	return;
      }

      const GlobalPoint& position = thisCell->getPosition();
     
      const TruncatedPyramid* pyr 
	= dynamic_cast< const TruncatedPyramid* > (thisCell);
      
      
      GlobalPoint axis;
      if( pyr ) {
	axis = pyr->getPosition(1);
      }
      
    
      reco::PFRecHit *rh = new reco::PFRecHit( detid.rawId(), PFLayer::ECAL_BARREL, energy, 
					       position.x(), position.y(), position.z(), 
					       axis.x(), axis.y(), axis.z() );
      
      ecalrechits.insert( make_pair(detid.rawId(), rh) ); 
    }

    // process ecal endcap rechits

    try {
      iEvent.getByLabel("ecalrechit", "EcalRecHitsEE", rhcHandle);
      cerr<<"got handle"<<endl;
      if (!(rhcHandle.isValid())) {
	cout<<"could not get a handle on EcalRecHitsEE!"<<endl;
	return;
      }
    }
    catch ( cms::Exception& ex ) {
      edm::LogError("PFClusterProducerError")
	<<"Error! can't get the EE rechits "<<ex.what()<<endl;
      return;
    }
    
    // cout<<"process endcap rechits"<<endl;
    for(unsigned i=0; i<rhcHandle->size(); i++) {
      
      double energy = (*rhcHandle)[i].energy();
      
      if(energy < threshEcalEndcap_ ) continue;
      
      const DetId& detid = (*rhcHandle)[i].detid();
      const CaloCellGeometry *thisCell = endcapGeometry->getGeometry(detid);
      
      if(!thisCell) {
	cerr<<"warning detid "<<detid.rawId()<<" not found in endcap geometry"<<endl;
	return;
      }
      
      const GlobalPoint& position = thisCell->getPosition();
      
      const TruncatedPyramid* pyr 
	= dynamic_cast< const TruncatedPyramid* > (thisCell);
      
      
      GlobalPoint axis;
      if( pyr ) {
	axis = pyr->getPosition(1);
      }
      
      
      reco::PFRecHit *rh = new reco::PFRecHit( detid.rawId(),  PFLayer::ECAL_ENDCAP, energy, 
					       position.x(), position.y(), position.z(), 
					       axis.x(), axis.y(), axis.z() );
      
      ecalrechits.insert( make_pair(detid.rawId(), rh) ); 
    }


    // find rechits neighbours
    
    // cout<<"find rechits neighbours"<<endl;
    for( PFClusterAlgo::IDH ih = ecalrechits.begin(); ih != ecalrechits.end(); ih++) {
      FindRecHitNeighbours( ih->second, ecalrechits, 
			    ecalBarrelTopology, 
			    *ecalBarrelGeometry, 
			    endcapTopology,
			    *endcapGeometry);
    }

    // cout<<"perform clustering"<<endl;
    PFClusterAlgo clusteralgo; 
    
    clusteralgo.SetThreshEcalBarrel( threshEcalBarrel_ );
    clusteralgo.SetThreshSeedEcalBarrel( threshSeedEcalBarrel_ );
    
    clusteralgo.SetThreshEcalEndcap( threshEcalEndcap_ );
    clusteralgo.SetThreshSeedEcalEndcap( threshSeedEcalEndcap_ );
    
    clusteralgo.Init( ecalrechits ); 
    clusteralgo.AllClusters();
    
    const map<unsigned, reco::PFRecHit* >& algohits = clusteralgo.GetIdRecHits();
    for(PFClusterAlgo::IDH ih=algohits.begin(); ih!=algohits.end(); ih++) {
      result->push_back( reco::PFRecHit( *(ih->second) ) );    
    }

    // clear all 
    // cout<<"clearing"<<endl;
    for( PFClusterAlgo::IDH ih = ecalrechits.begin(); ih != ecalrechits.end(); ih++) {  
      delete ih->second;
    }

  }
  
  if( processHcal_ ) {
    
    map<unsigned,  reco::PFRecHit* > hcalrechits;

    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<IdealGeometryRecord>().get(geoHandle);
    
    // get the hcalBarrel geometry
    const CaloSubdetectorGeometry *hcalBarrelGeometry = 
      geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    
    // get the hcal topology
    HcalTopology hcalTopology;
    
    // get the endcap geometry
    const CaloSubdetectorGeometry *hcalEndcapGeometry = 
      geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

    // HCAL rechits 
    vector<edm::Handle<HBHERecHitCollection> > hcalHandles;  
    try {
      iEvent.getManyByType(hcalHandles);
    } catch (...) {
      cout << "could not get handles on HBHERecHits !" << endl;
      return;
    }
    
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
	    pfrechit = CreateHcalRecHit(detid, 
					energy, 
					PFLayer::HCAL_BARREL1, 
					hcalBarrelGeometry );
	  }
	  break;
	case HcalEndcap:
	  if(energy > threshHcalEndcap_){
	    pfrechit = CreateHcalRecHit(detid, 
					energy, 
					PFLayer::HCAL_ENDCAP, 
					hcalEndcapGeometry );
	  }
	  break;
	default:
	  cerr<<"HCAL rechit: unknown layer !"<<endl;
	  continue;
	} 

	if(pfrechit) {
	  // cout<<(*pfrechit)<<endl;

	  //	  const math::XYZPoint& cpos = pfrechit->GetPositionXYZ();
	  // if( fabs(cpos.Eta() )< 0.06  )
	    hcalrechits.insert( make_pair(detid.rawId(), pfrechit) ); 
	}
      }
  
      cout<<"find HCAL neighbours"<<endl;
      for( PFClusterAlgo::IDH ih = hcalrechits.begin(); ih != hcalrechits.end(); ih++) {
	FindRecHitNeighbours( ih->second, hcalrechits, 
			      hcalTopology, 
			      *hcalBarrelGeometry, 
			      hcalTopology,
			      *hcalEndcapGeometry);
      }
      cout<<"start clustering"<<endl;

      // cout<<"perform clustering"<<endl;
      PFClusterAlgo clusteralgo; 
      
      clusteralgo.SetThreshHcalBarrel( threshHcalBarrel_ );
      clusteralgo.SetThreshSeedHcalBarrel( threshSeedHcalBarrel_ );
      
      clusteralgo.SetThreshHcalEndcap( threshHcalEndcap_ );
      clusteralgo.SetThreshSeedHcalEndcap( threshSeedHcalEndcap_ );
    
      clusteralgo.Init( hcalrechits ); 
      clusteralgo.AllClusters();

      cout<<"store hcal rechits"<<endl;
      const map<unsigned, reco::PFRecHit* >& algohits = clusteralgo.GetIdRecHits();
      for(PFClusterAlgo::IDH ih=algohits.begin(); ih!=algohits.end(); ih++) {
	result->push_back( reco::PFRecHit( *(ih->second) ) );    
      }

      // clear all 
      // cout<<"clearing"<<endl;
      for( PFClusterAlgo::IDH ih = hcalrechits.begin(); ih != hcalrechits.end(); ih++) {
	
	delete ih->second;
      }
    }
  }

  if(processPS_) {

    map<unsigned,  reco::PFRecHit* > psrechits;

    // get the ps geometry
    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<IdealGeometryRecord>().get(geoHandle);
    
    const CaloSubdetectorGeometry *psGeometry = 
      geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
    
    // get the ps topology
    EcalPreshowerTopology psTopology(geoHandle);

//     edm::ESHandle<CaloTopology> topoHandle;
//     iSetup.get<CaloTopologyRecord>().get(topoHandle);     

//     const CaloSubdetectorTopology *psTopology = 
//       topoHandle->getSubdetectorTopology(DetId::Ecal,EcalPreshower);


    // process rechits
    Handle< EcalRecHitCollection >   pRecHits;

    try {
      iEvent.getByLabel( "esrechit", "EcalRecHitsES", pRecHits);
      if (!(pRecHits.isValid())) {
	cout<<"could not get a handle on the EcalRecHitCollection!" 
	    <<endl;
	return;
      }
    } catch ( cms::Exception& ex ) {
      edm::LogError("PFClusterProducer") 
	<<"Error! can't get the preshower rechits"<<endl;
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
 
      cout<<"ps hit "<<hit<<" plane "<<detid.plane()<<endl;
      CaloNavigator<ESDetId> navigator(detid,&psTopology);

      ESDetId n = navigator.north();
      if(n != DetId(0) )
	cout<<"north "<<n.plane()<<endl;
      navigator.home();
     
      ESDetId s = navigator.south();
      if(s != DetId(0) )
	cout<<"south "<<s.plane()<<endl;
      navigator.home();
     
      if(!thisCell) {
	cerr<<"warning detid "<<detid.rawId()
	    <<" not found in preshower geometry"<<endl;
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
	cerr<<"incorrect preshower plane !! plane number "<<detid.plane()<<endl;
	assert(0);
      }
 
      reco::PFRecHit *pfrechit = new reco::PFRecHit( detid.rawId(), layer, energy, 
						  position.x(), position.y(), position.z(), 
						  0,0,0 );

      psrechits.insert( make_pair(detid.rawId(), pfrechit) );
 
    }
    
    // cout<<"find rechits neighbours"<<endl;
    for( PFClusterAlgo::IDH ih = psrechits.begin(); ih != psrechits.end(); ih++) {
      FindRecHitNeighbours( ih->second, psrechits, 
			    psTopology, 
			    *psGeometry, 
			    psTopology,
			    *psGeometry);
    }

    // cout<<"perform clustering"<<endl;
    PFClusterAlgo clusteralgo; 
    
    clusteralgo.SetThreshPS( threshPS_ );
    clusteralgo.SetThreshSeedPS( threshSeedPS_ );
       
    clusteralgo.Init( psrechits ); 
    clusteralgo.AllClusters();
    
    cout<<"store hcal rechits"<<endl;
    const map<unsigned, reco::PFRecHit* >& algohits = clusteralgo.GetIdRecHits();
    for(PFClusterAlgo::IDH ih=algohits.begin(); ih!=algohits.end(); ih++) {
      result->push_back( reco::PFRecHit( *(ih->second) ) );    
    }

    // clear all 
    // cout<<"clearing"<<endl;
    for( PFClusterAlgo::IDH ih = psrechits.begin(); ih != psrechits.end(); ih++) {  
      delete ih->second;
    }
  }
  
  
  iEvent.put( result );
 

  /* This is an event example
  //Read 'ExampleData' from the Event
  Handle<ExampleData> pIn;
  iEvent.getByLabel("example",pIn);

  //Use the ExampleData to create an ExampleData2 which 
  // is put into the Event
  auto_ptr<ExampleData2> pOut(new ExampleData2(*pIn));
  iEvent.put(pOut);
  */

  /* this is an EventSetup example
  //Read SetupData from the SetupRecord in the EventSetup
  ESHandle<SetupData> pSetup;
  iSetup.get<SetupRecord>().get(pSetup);
  */
}

void 
PFClusterProducer::FindRecHitNeighbours( reco::PFRecHit* rh, 
					 const map<unsigned,  reco::PFRecHit* >& rechits, 
					 const CaloSubdetectorTopology& barrelTopology, 
					 const CaloSubdetectorGeometry& barrelGeometry, 
					 const CaloSubdetectorTopology& endcapTopology, 
					 const CaloSubdetectorGeometry& endcapGeometry ) {
  
  
  // cerr<<"find neighbours "<<endl;

  const math::XYZPoint& cpos = rh->GetPositionXYZ();
  double posx = cpos.X();
  double posy = cpos.Y();
  double posz = cpos.Z();

  bool debug = false;
  if( rh->GetLayer() == PFLayer::PS1 ||
      rh->GetLayer() == PFLayer::PS2 ) debug = true;
  

  DetId detid( rh->GetDetId() );
  // if(debug) cerr<<"detid created "<<endl;

  CaloNavigator<DetId>* navigator = 0;
  CaloSubdetectorGeometry* geometry = 0;
  CaloSubdetectorGeometry* othergeometry = 0;



  if(debug) cerr<<"find hcal neighbours "<<rh->GetLayer()<<endl;
  
  switch( rh->GetLayer()  ) {
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
    othergeometry = const_cast< CaloSubdetectorGeometry* > (&barrelGeometry);
    break;
  case PFLayer::HCAL_BARREL1:
    navigator = new CaloNavigator<DetId>(detid, &barrelTopology);
    geometry = const_cast< CaloSubdetectorGeometry* > (&barrelGeometry);
    othergeometry = const_cast< CaloSubdetectorGeometry* > (&endcapGeometry);
    break;
  case PFLayer::PS1:
  case PFLayer::PS2:
    navigator = new CaloNavigator<DetId>(detid, &barrelTopology);
    geometry = const_cast< CaloSubdetectorGeometry* > (&barrelGeometry);
    othergeometry = const_cast< CaloSubdetectorGeometry* > (&endcapGeometry);
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

      ESDetId esid(northeast.rawId());
      cout<<"nb layer : "<<esid.plane()<<endl;

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
	
	rh->SetNECorner( cposx, cposy, cposz );
      }
      else if(debug) cerr<<cpos.Eta()<<" "<<cpos.Phi()
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
	
	rh->SetSWCorner( cposx, cposy, cposz );
      }
      else if(debug) cerr<<cpos.Eta()<<" "<<cpos.Phi()
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
      
	rh->SetSECorner( cposx, cposy, cposz );
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
      
	rh->SetNWCorner( cposx, cposy, cposz );
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
    
  rh->SetNeighbours( neighbours );

  cout<<(*rh)<<endl;

}


reco::PFRecHit* PFClusterProducer::CreateHcalRecHit( const DetId& detid,
						     double energy,
						     int layer,
						     const CaloSubdetectorGeometry* geom ) {
  
  const CaloCellGeometry *thisCell = geom->getGeometry(detid);
  if(!thisCell) {
    cerr<<"warning detid "<<detid.rawId()<<" not found in hcal geometry"<<endl;
    return 0;
  }
      
  const GlobalPoint& position = thisCell->getPosition();
  
  reco::PFRecHit *rh = new reco::PFRecHit( detid.rawId(),  layer, energy, 
					   position.x(), position.y(), position.z(), 
					   0,0,0 );

  return rh;
}



//define this as a plug-in
DEFINE_FWK_MODULE(PFClusterProducer)
