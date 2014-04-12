#include "Calibration/EcalAlCaRecoProducers/interface/AlCaECALRecHitReducer.h"
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
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

//#define ALLrecHits
//#define DEBUG

//#define QUICK -> if commented loop over the recHits of the SC and add them to the list of recHits to be saved
//                 comment it if you want a faster module but be sure the window is large enough

AlCaECALRecHitReducer::AlCaECALRecHitReducer(const edm::ParameterSet& iConfig) 
{

  ebRecHitsLabel_ = iConfig.getParameter< edm::InputTag > ("ebRecHitsLabel");
  eeRecHitsLabel_ = iConfig.getParameter< edm::InputTag > ("eeRecHitsLabel");
  //  esRecHitsLabel_ = iConfig.getParameter< edm::InputTag > ("esRecHitsLabel");
  electronLabel_ = iConfig.getParameter< edm::InputTag > ("electronLabel");

  alcaBarrelHitsCollection_ = iConfig.getParameter<std::string>("alcaBarrelHitCollection");
  alcaEndcapHitsCollection_ = iConfig.getParameter<std::string>("alcaEndcapHitCollection");
  //  alcaPreshowerHitsCollection_ = iConfig.getParameter<std::string>("alcaPreshowerHitCollection");
  
  etaSize_ = iConfig.getParameter<int> ("etaSize");
  phiSize_ = iConfig.getParameter<int> ("phiSize");
  // FIXME: minimum size of etaSize_ and phiSize_
  if ( phiSize_ % 2 == 0 ||  etaSize_ % 2 == 0)
    edm::LogError("AlCaECALRecHitReducerError") << "Size of eta/phi should be odd numbers";
 
  weight_= iConfig.getParameter<double> ("eventWeight");
 
  //  esNstrips_  = iConfig.getParameter<int> ("esNstrips");
  //  esNcolumns_ = iConfig.getParameter<int> ("esNcolumns");
  
  //register your products
  produces< EBRecHitCollection > (alcaBarrelHitsCollection_) ;
  produces< EERecHitCollection > (alcaEndcapHitsCollection_) ;
  //  produces< ESRecHitCollection > (alcaPreshowerHitsCollection_) ;
  produces< double > ("weight") ;
}


AlCaECALRecHitReducer::~AlCaECALRecHitReducer()
{}


// ------------ method called to produce the data  ------------
void
AlCaECALRecHitReducer::produce (edm::Event& iEvent, 
                                const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  
  EcalRecHitCollection::const_iterator recHit_itr;

  // get the ECAL geometry:
  ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);

  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  const CaloTopology *caloTopology = theCaloTopology.product();
  
  // Get GSFElectrons
  Handle<reco::GsfElectronCollection> pElectrons;
  iEvent.getByLabel(electronLabel_, pElectrons);
  if (!pElectrons.isValid()) {
    edm::LogError ("reading") << electronLabel_ << " not found" ; 
    //      std::cerr << "[AlCaECALRecHitReducer]" << electronLabel_ << " not found" ; 
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
  
  //   // get ES RecHits
  //   Handle<ESRecHitCollection> preshowerRecHitsHandle;
  //   bool preshowerIsFull = true ;
  
  //   iEvent.getByLabel(esRecHitsLabel_,preshowerRecHitsHandle);
  //   if (!preshowerRecHitsHandle.isValid()) {
  //     edm::LogError ("reading") << esRecHitsLabel_ << " not found" ; 
  //     preshowerIsFull = false ;
  //   }
  
  //   const ESRecHitCollection * preshowerHitsCollection = 0 ;
  //   if (preshowerIsFull)  
  //     preshowerHitsCollection = preshowerRecHitsHandle.product () ;

  //   // make a vector to store the used ES rechits:
  //   set<ESDetId> used_strips;
  //   used_strips.clear();
 
  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > miniEBRecHitCollection (new EBRecHitCollection) ;
  std::auto_ptr< EERecHitCollection > miniEERecHitCollection (new EERecHitCollection) ;  
  //  std::auto_ptr< ESRecHitCollection > miniESRecHitCollection (new ESRecHitCollection) ;  
  std::auto_ptr< double > weight (new double(1));
  (*weight) = weight_;
  
  //  loop on SiStrip Electrons
  
  reco::GsfElectronCollection::const_iterator eleIt;
  int nEle_EB=0;
  int nEle_EE=0;

  std::set<DetId> reducedRecHit_EBmap;
  std::set<DetId> reducedRecHit_EEmap;

  for (eleIt=electronCollection->begin(); eleIt!=electronCollection->end(); eleIt++) {
    // barrel
    const reco::SuperCluster& sc = *(eleIt->superCluster()) ;

    if (eleIt->isEB()) {
      nEle_EB++;

      // find the seed 
      EBDetId seed=(sc.seed()->seed());

      std::vector<DetId> recHit_window = caloTopology->getWindow(seed, phiSize_, etaSize_);
      for(unsigned int i =0; i < recHit_window.size(); i++){
#ifdef DEBUG2
	std::cout << i << "/" << recHit_window.size() << "\t" << recHit_window[i]() << std::endl;
#endif
	reducedRecHit_EBmap.insert(recHit_window[i]);
#ifdef DEBUG2
	EBDetId ebrechit(recHit_window[i]);
	std::cout << ebrechit.ieta() << "\t" << ebrechit.iphi() << std::endl;
#endif
      }
      
#ifdef DEBUG      
      // find the most energetic crystal 
      float energy_recHit_max=-999;

      if(reducedRecHit_EBmap.size() < sc.size())
	std::cerr << "[WARNING] number of recHit in selected window < RECO SC recHits!" << std::endl;
#endif

#ifndef QUICK
      const std::vector< std::pair<DetId, float> > & scHits = sc.hitsAndFractions();

#ifdef DEBUG
      std::vector< std::pair<DetId, float> >::const_iterator scHit_max_itr = scHits.end();
#endif
      for(std::vector< std::pair<DetId, float> >::const_iterator scHit_itr = scHits.begin(); 
	  scHit_itr != scHits.end(); scHit_itr++){
	// the map fills just one time (avoiding double insert of recHits)
	reducedRecHit_EBmap.insert(scHit_itr->first);

#ifdef DEBUG2
	const EcalRecHit ecalRecHit = *(barrelHitsCollection->find( (*scHit_itr).first ));
	if(energy_recHit_max <  ecalRecHit.energy()){
	  scHit_max_itr = scHit_itr;
	  energy_recHit_max=ecalRecHit.energy();
	}
#endif
      }
#endif

#ifdef DEBUG2
      // cross check, the seed should be the highest energetic crystal in the SC
      if(EBDetId(scHit_max_itr->first) != seed)
	std::cerr << "[ERROR] highest energetic crystal is not the seed of the SC" << std::endl;

      else{
	
	std::cout << "[DEBUG] highest energetic crystal = " << EBDetId(scHit_max_itr->first) << std::endl;
	std::cout << "[DEBUG] seed of the SC = " << seed << std::endl;
      }
#endif
      //                                                                   (id, phi, eta)

      if(reducedRecHit_EBmap.size() < sc.size()){
	if(eleIt->ecalDrivenSeed())
	  edm::LogError("AlCaSavedRecHitsEB") << "[ERROR] ecalDrivenSeed: number of saved recHits < RECO SC recHits!: " << reducedRecHit_EBmap.size() << " < " << sc.size() << std::endl;
	else
	  edm::LogWarning("AlCaSavedRecHitsEB") << "[WARNING] trackerDrivenSeed: number of saved recHits < RECO SC recHits!: " << reducedRecHit_EBmap.size() << " < " << sc.size() << std::endl;

      }

    } else { // endcap
      nEle_EE++;

      // find the seed 
      EEDetId seed=(sc.seed()->seed());

      // get detId for a window around the seed of the SC
      int sideSize = std::max(phiSize_,etaSize_);
      std::vector<DetId> recHit_window = caloTopology->getWindow(seed, sideSize, sideSize);

      // fill the recHit map with the DetId of the window
      for(std::vector<DetId>::const_iterator window_itr = recHit_window.begin(); 
	  window_itr != recHit_window.end(); window_itr++){
	reducedRecHit_EEmap.insert(*window_itr);
      }
#ifdef DEBUG
      if(reducedRecHit_EEmap.size() < sc.size())
	std::cerr << "[WARNING] number of recHit in selected window < RECO SC recHits!" << std::endl;
#endif

      const std::vector< std::pair<DetId, float> > & scHits = sc.hitsAndFractions();

#ifndef QUICK
      // fill the recHit map with the DetId of the SC recHits

      for(std::vector< std::pair<DetId, float> >::const_iterator scHit_itr = scHits.begin(); 
	  scHit_itr != scHits.end(); scHit_itr++){
	// the map fills just one time (avoiding double insert of recHits)
	reducedRecHit_EEmap.insert(scHit_itr->first);	

      }
#endif

      if(reducedRecHit_EEmap.size() < sc.size()){
	if(eleIt->ecalDrivenSeed())
	  edm::LogError("AlCaSavedRecHitsEE") << "[ERROR] ecalDrivenSeed: number of saved recHits < RECO SC recHits!: " << reducedRecHit_EEmap.size() << " < " << sc.size() << std::endl;
	else
	  edm::LogWarning("AlCaSavedRecHitsEE") << "[WARNING] trackerDrivenSeed: number of saved recHits < RECO SC recHits!: " << reducedRecHit_EEmap.size() << " < " << sc.size() << std::endl;

      }

    } // end of endcap
  }

#ifndef ALLrecHits
  for(std::set<DetId>::const_iterator itr = reducedRecHit_EBmap.begin();
      itr != reducedRecHit_EBmap.end(); itr++){
    if (barrelHitsCollection->find(*itr) != barrelHitsCollection->end())
      miniEBRecHitCollection->push_back(*(barrelHitsCollection->find(*itr)));
  }
#else
  for(EcalRecHitCollection::const_iterator recHits_itr = barrelHitsCollection->begin();
      recHits_itr != barrelHitsCollection->end();
      recHits_itr++){
    miniEBRecHitCollection->push_back(*recHits_itr);
  }
#endif

  // fill the alcareco reduced recHit collection
  for(std::set<DetId>::const_iterator itr = reducedRecHit_EEmap.begin();
      itr != reducedRecHit_EEmap.end(); itr++){
    if (endcapHitsCollection->find(*itr) != endcapHitsCollection->end())
      miniEERecHitCollection->push_back(*(endcapHitsCollection->find(*itr)));
  }



#ifdef DEBUG
  std::cout << "nEle_EB= " << nEle_EB << "\tnEle_EE = " << nEle_EE << std::endl;
  if(nEle_EB > 0 && miniEBRecHitCollection->size() < (unsigned int) phiSize_*etaSize_)
    edm::LogError("AlCaECALRecHitReducerError") << "Size EBRecHitCollection < " << phiSize_*etaSize_ << ": "  << miniEBRecHitCollection->size() ;

  int side = phiSize_ ;
  if (phiSize_ < etaSize_) side = etaSize_ ;

  if(nEle_EE > 0 && miniEERecHitCollection->size() < (unsigned int )side*side)
    edm::LogError("AlCaECALRecHitReducerError") << "Size EERecHitCollection < " << side*side << ": "  << miniEERecHitCollection->size() ;
#endif
  
  //Put selected information in the event
  iEvent.put( miniEBRecHitCollection,alcaBarrelHitsCollection_ );
  iEvent.put( miniEERecHitCollection,alcaEndcapHitsCollection_ );     
  //  iEvent.put( miniESRecHitCollection,alcaPreshowerHitsCollection_ );     
  iEvent.put( weight, "weight");     
}


DEFINE_FWK_MODULE(AlCaECALRecHitReducer);

