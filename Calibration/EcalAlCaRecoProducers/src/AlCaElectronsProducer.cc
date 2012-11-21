#include "Calibration/EcalAlCaRecoProducers/interface/AlCaElectronsProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

AlCaElectronsProducer::AlCaElectronsProducer(const edm::ParameterSet& iConfig) 
{

  ebRecHitsLabel_ = iConfig.getParameter< edm::InputTag > ("ebRecHitsLabel");
  eeRecHitsLabel_ = iConfig.getParameter< edm::InputTag > ("eeRecHitsLabel");
  esRecHitsLabel_ = iConfig.getParameter< edm::InputTag > ("esRecHitsLabel");
  electronLabel_ = iConfig.getParameter< edm::InputTag > ("electronLabel");

  alcaBarrelHitsCollection_ = iConfig.getParameter<std::string>("alcaBarrelHitCollection");
  alcaEndcapHitsCollection_ = iConfig.getParameter<std::string>("alcaEndcapHitCollection");
  alcaPreshowerHitsCollection_ = iConfig.getParameter<std::string>("alcaPreshowerHitCollection");
  
  etaSize_ = iConfig.getParameter<int> ("etaSize");
  phiSize_ = iConfig.getParameter<int> ("phiSize");
  if ( phiSize_ % 2 == 0 ||  etaSize_ % 2 == 0)
    edm::LogError("AlCaElectronsProducerError") << "Size of eta/phi should be odd numbers";
 
  weight_= iConfig.getParameter<double> ("eventWeight");
 
  esNstrips_  = iConfig.getParameter<int> ("esNstrips");
  esNcolumns_ = iConfig.getParameter<int> ("esNcolumns");
  
   //register your products
  produces< EBRecHitCollection > (alcaBarrelHitsCollection_) ;
  produces< EERecHitCollection > (alcaEndcapHitsCollection_) ;
  produces< ESRecHitCollection > (alcaPreshowerHitsCollection_) ;
  produces< double > ("weight") ;
}


AlCaElectronsProducer::~AlCaElectronsProducer()
{}


// ------------ method called to produce the data  ------------
void
AlCaElectronsProducer::produce (edm::Event& iEvent, 
                                const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  
  // get the ECAL geometry:
  ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
  const CaloSubdetectorGeometry *geometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  const CaloSubdetectorGeometry *& geometry_p = geometry;
 
  // Get GSFElectrons
  Handle<reco::GsfElectronCollection> pElectrons;
  iEvent.getByLabel(electronLabel_, pElectrons);
  if (!pElectrons.isValid()) {
    edm::LogError ("reading") << electronLabel_ << " not found" ; 
    //      std::cerr << "[AlCaElectronsProducer]" << electronLabel_ << " not found" ; 
    return ;
  }
  
  const reco::GsfElectronCollection * electronCollection = 
    pElectrons.product();
  
  // get RecHits
  Handle<EBRecHitCollection> barrelRecHitsHandle;
  bool barrelIsFull = true ;
  
  iEvent.getByLabel(ebRecHitsLabel_,barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    edm::LogError ("reading") << ebRecHitsLabel_ << " not found" ; 
    barrelIsFull = false ;
  }
  
  const EBRecHitCollection * barrelHitsCollection = 0 ;
  if (barrelIsFull)  
    barrelHitsCollection = barrelRecHitsHandle.product () ;
  
  // get RecHits
  Handle<EERecHitCollection> endcapRecHitsHandle;
  bool endcapIsFull = true ;
  
  iEvent.getByLabel(eeRecHitsLabel_,endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    edm::LogError ("reading") << eeRecHitsLabel_ << " not found" ; 
    endcapIsFull = false ;
  }
  
  const EERecHitCollection * endcapHitsCollection = 0 ;
  if (endcapIsFull)  
    endcapHitsCollection = endcapRecHitsHandle.product () ;
  //  const EERecHitCollection * endcapHitsCollection = endcapRecHitsHandle.product();
  
  // get ES RecHits
  Handle<ESRecHitCollection> preshowerRecHitsHandle;
  bool preshowerIsFull = true ;
  
  iEvent.getByLabel(esRecHitsLabel_,preshowerRecHitsHandle);
  if (!preshowerRecHitsHandle.isValid()) {
    edm::LogError ("reading") << esRecHitsLabel_ << " not found" ; 
    preshowerIsFull = false ;
  }
  
  const ESRecHitCollection * preshowerHitsCollection = 0 ;
  if (preshowerIsFull)  
    preshowerHitsCollection = preshowerRecHitsHandle.product () ;

  // make a vector to store the used ES rechits:
  set<ESDetId> used_strips;
  used_strips.clear();
 
  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > miniEBRecHitCollection (new EBRecHitCollection) ;
  std::auto_ptr< EERecHitCollection > miniEERecHitCollection (new EERecHitCollection) ;  
  std::auto_ptr< ESRecHitCollection > miniESRecHitCollection (new ESRecHitCollection) ;  
  std::auto_ptr< double > weight (new double(1));
  (*weight) = weight_;
  
  //  loop on SiStrip Electrons
  
  reco::GsfElectronCollection::const_iterator eleIt;
  int ii=0;
  
  for (eleIt=electronCollection->begin(); eleIt!=electronCollection->end(); eleIt++) {
    //PG barrel
    if (fabs(eleIt->eta()) <= 1.479) {
      
      ii++;         
      const reco::SuperCluster& sc = *(eleIt->superCluster()) ;
      
      int yy = 0;
      double currEnergy = 0.;
      EBDetId maxHit(0);
      std::vector<EBDetId> scXtals;
      scXtals.clear();
      const std::vector< std::pair<DetId, float> > & v1 = sc.hitsAndFractions();
      
      for(std::vector< std::pair<DetId, float> >::const_iterator idsIt = v1.begin(); 
	  idsIt != v1.end(); ++idsIt) 
	{
	  yy++;
	  
	  //PG discard hits not belonging to the Ecal Barrel
	  if((*idsIt).first.subdetId()!=EcalBarrel || (*idsIt).first.det()!= DetId::Ecal) continue;
	  
	  if((barrelHitsCollection->find( (*idsIt).first ))->energy() > currEnergy) {
            currEnergy=(barrelHitsCollection->find( (*idsIt).first ))->energy();
            maxHit=(*idsIt).first;
          }
	  miniEBRecHitCollection->push_back(*(barrelHitsCollection->find( (*idsIt).first )));
	  scXtals.push_back( (*idsIt).first );
	}
      
      if (!maxHit.null())
	for (int icry=0;icry< etaSize_*phiSize_;icry++)
	  {
	    
	    unsigned int row = icry / etaSize_;
	    unsigned int column= icry % etaSize_;
	    int curr_eta=maxHit.ieta() + column - (etaSize_/2);
	    int curr_phi=maxHit.iphi() + row - (phiSize_/2);
	    
	    if (curr_eta * maxHit.ieta() <= 0) {if (maxHit.ieta() > 0) curr_eta--; else curr_eta++; }  // JUMP over 0
	    if (curr_phi < 1) curr_phi += 360;
	    if (curr_phi > 360) curr_phi -= 360;
	    if (!(EBDetId::validDetId(curr_eta,curr_phi))) continue; 
	    EBDetId det = EBDetId(curr_eta,curr_phi,EBDetId::ETAPHIMODE);
	    std::vector<EBDetId>::const_iterator usedIds;
	    
	    bool HitAlreadyUsed=false;
	    for(usedIds=scXtals.begin(); usedIds!=scXtals.end(); usedIds++)
	      if(*usedIds==det)
		{
		  HitAlreadyUsed=true;
		  break;
		}
	    
	    if(!HitAlreadyUsed)
	      if (barrelHitsCollection->find(det) != barrelHitsCollection->end())
		miniEBRecHitCollection->push_back(*(barrelHitsCollection->find(det)));
	  }
    } //PG barrel
    else //PG endcap
      {
	
	ii++;         
	const reco::SuperCluster& sc = *(eleIt->superCluster()) ;
	
	int yy = 0;
	double currEnergy = 0.;
	EEDetId maxHit(0);
	vector<EEDetId> scXtals;
	scXtals.clear();
	const std::vector< std::pair<DetId, float> > & v1 = sc.hitsAndFractions();
	
	for(std::vector< std::pair<DetId, float> >::const_iterator idsIt = v1.begin(); 
	    idsIt != v1.end(); ++idsIt) 
	  {
	    yy++;
	    
	    //PG discard hits belonging to the Ecal Barrel
	    if((*idsIt).first.subdetId()!=EcalEndcap || 
	       (*idsIt).first.det()!= DetId::Ecal) continue;
	    
	    if((endcapHitsCollection->find( (*idsIt).first ))->energy() > currEnergy) 
	      {
		currEnergy=(endcapHitsCollection->find( (*idsIt).first ))->energy();
		maxHit=(*idsIt).first;
	      }
	    miniEERecHitCollection->push_back (
					       *(endcapHitsCollection->find ( (*idsIt).first ))
					       );
	    scXtals.push_back ( (*idsIt).first ) ; 
	  }  
	
	int side = phiSize_ ;
	if (phiSize_ < etaSize_) side = etaSize_ ;
	int iz = 1 ;
	if (eleIt->eta () < 0)  iz = -1 ;
	if (!maxHit.null())
	  //PG loop over the local array of xtals
	  for (int icry = 0 ; icry < side*side ; icry++)
	    {          
	      unsigned int row = icry / side ;
	      unsigned int column = icry % side ;
	      int curr_eta = maxHit.ix () + column - (side/2);
	      int curr_phi = maxHit.iy () + row - (side/2);
	      if (   curr_eta <= 0 || curr_phi <= 0
		     || curr_eta > 100 || curr_phi > 100 ) continue ;
	      if (!(EEDetId::validDetId(curr_eta,curr_phi,iz))) continue;
              EEDetId det = EEDetId (curr_eta,curr_phi,iz,EEDetId::XYMODE) ; 
              if (find (scXtals.begin (), scXtals.end (), det) != scXtals.end ())
                if (endcapHitsCollection->find (det) != endcapHitsCollection->end ())
                  miniEERecHitCollection->push_back (*(endcapHitsCollection->find (det))) ;
	    }
	
	// Loop over basic clusters to find ES rec hits
	reco::CaloCluster_iterator bc_iter = sc.clustersBegin();
	for ( ; bc_iter != sc.clustersEnd(); ++bc_iter ) {  
	  if (geometry) {
	    double X = (*bc_iter)->x();
	    double Y = (*bc_iter)->y();
	    double Z = (*bc_iter)->z();        
	    const GlobalPoint point(X,Y,Z);    
	    
	    DetId tmp1 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(point, 1);
	    DetId tmp2 = (dynamic_cast<const EcalPreshowerGeometry*>(geometry_p))->getClosestCellInPlane(point, 2);
	    ESDetId strip1 = (tmp1 == DetId(0)) ? ESDetId(0) : ESDetId(tmp1);
	    ESDetId strip2 = (tmp2 == DetId(0)) ? ESDetId(0) : ESDetId(tmp2);     

	    int esindexE1 = (tmp1 == DetId(0)) ? -1 : (strip1.six()-1)*32+strip1.strip();
	    int esindexE2 = (tmp2 == DetId(0)) ? -1 : (strip2.siy()-1)*32+strip2.strip();
	    int esindexRh = 0; 
	    
	    EcalRecHitCollection::const_iterator esit;
	    for (esit = preshowerHitsCollection->begin(); esit != preshowerHitsCollection->end(); esit++) {

	      ESDetId esid = ESDetId(esit->id());

	      if (used_strips.find(esid) != used_strips.end()) continue;

	      if (strip1 != ESDetId(0) && esid.plane()==1) {
	       
		esindexRh = (esid.six()-1)*32+esid.strip();		
		
		if (fabs(esindexE1-esindexRh) <= esNstrips_ && fabs(strip1.siy()-esid.siy()) <= esNcolumns_ && strip1.zside()==esid.zside()) {
		  miniESRecHitCollection->push_back(*esit);
		  used_strips.insert(esid);
		}
	      }
	      if (strip2 != ESDetId(0) && esid.plane()==2) {

		esindexRh = (esid.siy()-1)*32+esid.strip();

		if (fabs(esindexE2-esindexRh) <= esNstrips_ && fabs(strip2.six()-esid.six()) <= esNcolumns_ && strip2.zside()==esid.zside()) {
		  miniESRecHitCollection->push_back(*esit);
		  used_strips.insert(esid);
		}
	      }

	    }
	  }
	}
	
      } //PG endcap
    
  } //PG loop on Si strip electrons
  
  //Put selected information in the event
  iEvent.put( miniEBRecHitCollection,alcaBarrelHitsCollection_ );
  iEvent.put( miniEERecHitCollection,alcaEndcapHitsCollection_ );     
  iEvent.put( miniESRecHitCollection,alcaPreshowerHitsCollection_ );     
  iEvent.put( weight, "weight");     
}
