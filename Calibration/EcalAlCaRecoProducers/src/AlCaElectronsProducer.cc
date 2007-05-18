#include "Calibration/EcalAlCaRecoProducers/interface/AlCaElectronsProducer.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"


AlCaElectronsProducer::AlCaElectronsProducer(const edm::ParameterSet& iConfig)
{
  ebRecHitsLabel_ = iConfig.getParameter< edm::InputTag > ("ebRecHitsLabel");
  electronLabel_ = iConfig.getParameter< edm::InputTag > ("electronLabel");

  alcaBarrelHitsCollection_ = iConfig.getParameter<std::string>("alcaBarrelHitCollection");
  
  etaSize_ = iConfig.getParameter<int> ("etaSize");
  phiSize_ = iConfig.getParameter<int> ("phiSize");
  if ( phiSize_ % 2 == 0 ||  etaSize_ % 2 == 0)
    edm::LogError("AlCaElectronsProducerError") << "Size of eta/phi should be odd numbers";
 
   //register your products
  produces< EBRecHitCollection >(alcaBarrelHitsCollection_);
  
}


AlCaElectronsProducer::~AlCaElectronsProducer()
{
}


// ------------ method called to produce the data  ------------
void
AlCaElectronsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

  // Get siStripElectrons
  Handle<reco::PixelMatchGsfElectronCollection> pElectrons;
  iEvent.getByLabel(electronLabel_, pElectrons);
  const reco::PixelMatchGsfElectronCollection* electronCollection = pElectrons.product();
  
  // get RecHits
  Handle<EBRecHitCollection> barrelRecHitsHandle;
  iEvent.getByLabel(ebRecHitsLabel_,barrelRecHitsHandle);
  const EBRecHitCollection* barrelHitsCollection = barrelRecHitsHandle.product();

  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > miniEBRecHitCollection( new EBRecHitCollection );

//  loop on SiStrip Electrons
  
  reco::PixelMatchGsfElectronCollection::const_iterator eleIt;
  int ii=0;
  
  for (eleIt=electronCollection->begin(); eleIt!=electronCollection->end(); eleIt++) {
    if (fabs(eleIt->eta()) <= 10){
      //1.479) 
      {
	
	ii++;         
	const reco::SuperCluster& sc = *(eleIt->superCluster()) ;
	
	int yy = 0;
	double currEnergy = 0.;
	EBDetId maxHit(0);
	vector<EBDetId> scXtals;
	scXtals.clear();
	const std::vector<DetId> & v1 = sc.getHitsByDetId();
	
	for( std::vector<DetId>::const_iterator idsIt = v1.begin(); idsIt != v1.end(); ++idsIt) {
	  yy++;
	  
	  if((*idsIt).subdetId()!=EcalBarrel || (*idsIt).det()!= DetId::Ecal) continue;
	  
	  if((barrelHitsCollection->find(*idsIt))->energy() > currEnergy) {
	    currEnergy=(barrelHitsCollection->find(*idsIt))->energy();
	    maxHit=*idsIt;
	  }
	  miniEBRecHitCollection->push_back(*(barrelHitsCollection->find(*idsIt)));
	  scXtals.push_back(*idsIt);
	}
	
	if (!maxHit.null())
	  for (unsigned int icry=0;icry< etaSize_*phiSize_;icry++)
	    {
	      
	      unsigned int row = icry / etaSize_;
	      unsigned int column= icry % etaSize_;
	      int curr_eta=maxHit.ieta() + column - (etaSize_/2);
	      int curr_phi=maxHit.iphi() + row - (phiSize_/2);
	      
	      if (curr_eta * maxHit.ieta() <= 0) {if (maxHit.ieta() > 0) curr_eta--; else curr_eta++; }  // JUMP over 0
	      if (curr_phi < 1) curr_phi += 360;
	      if (curr_phi > 360) curr_phi -= 360;
	      
	      try
		{
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
	      catch (...)
		{
		}
	    }
      }
  }
  
  
  //Put selected information in the event
  iEvent.put( miniEBRecHitCollection,alcaBarrelHitsCollection_ );
    
}
