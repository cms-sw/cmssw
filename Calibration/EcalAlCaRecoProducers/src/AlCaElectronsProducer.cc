#include "Calibration/EcalAlCaRecoProducers/interface/AlCaElectronsProducer.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

AlCaElectronsProducer::AlCaElectronsProducer(const edm::ParameterSet& iConfig)
{
//input collections
siStripElectronProducer_ = iConfig.getParameter< std::string > ("siStripElectronProducer");

siStripElectronCollection_ = iConfig.getParameter<std::string>("siStripElectronCollection");

BarrelHitsCollection_ = iConfig.getParameter< std::string > ("barrelHitCollection");

ecalHitsProducer_ = iConfig.getParameter< std::string > ("ecalRecHitsProducer");


//output collections


alcaSiStripElectronCollection_ = iConfig.getParameter<std::string>("alcaSiStripElectronCollection");

alcaBarrelHitsCollection_ = iConfig.getParameter<std::string>("alcaBarrelHitCollection");

alcaHybridSuperClusterCollection_ = iConfig.getParameter<std::string>("alcaHybridSuperClusterCollection");

//alcaCorrectedHybridSuperClusterCollection_ = iConfig.getParameter<std::string>("alcaCorrectedHybridSuperClusterCollection");

ptCut_ = iConfig.getParameter< double > ("ptCut");


   //register your products

   produces<reco::SiStripElectronCollection>(alcaSiStripElectronCollection_);

   produces< EBRecHitCollection >(alcaBarrelHitsCollection_);

   produces<reco::SuperClusterCollection>(alcaHybridSuperClusterCollection_);

//   produces<reco::SuperClusterCollection>(alcaCorrectedHybridSuperClusterCollection_);



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
  Handle<reco::SiStripElectronCollection> pSiStripElectrons;
  try {
     iEvent.getByLabel(siStripElectronProducer_, siStripElectronCollection_, pSiStripElectrons);
   } catch ( std::exception& ex ) {
     LogDebug("") << "AlCaElectronsProducer: Error! can't get product!" << std::endl;
   }
  const reco::SiStripElectronCollection* siStripElectronCollection = pSiStripElectrons.product();

// get RecHits
  Handle<EBRecHitCollection> barrelRecHitsHandle;
  try {
    iEvent.getByLabel(ecalHitsProducer_,BarrelHitsCollection_,barrelRecHitsHandle);
  } catch ( std::exception& ex ) {
    std::cout << "AlCaElectronsProducer: Error! can't get product!" << std::endl;
  }
  const EBRecHitCollection* barrelHitsCollection = barrelRecHitsHandle.product();



  //Create empty output collections

    std::auto_ptr<reco::SiStripElectronCollection> miniSiStripElectronCollection(new reco::SiStripElectronCollection);

    std::auto_ptr< EBRecHitCollection > miniEBRecHitCollection( new EBRecHitCollection );

    std::auto_ptr< reco::SuperClusterCollection > miniSuperClusterCollection( new reco::SuperClusterCollection );


//  loop on SiStrip Electrons

    reco::SiStripElectronCollection::const_iterator siStripEleIt;
    int ii=0;
 
    for (siStripEleIt=siStripElectronCollection->begin(); siStripEleIt!=siStripElectronCollection->end(); siStripEleIt++) {
      if (siStripEleIt->pt() >= ptCut_) {

        ii++;         
        
	 const reco::SuperCluster& sc = *(siStripEleIt->superCluster()) ;
	              
	       int yy=0;
               double mySCenergy=0.0;
    	       double currEnergy =0.;
 	       EBDetId maxHit;
    	       vector<EBDetId> scXtals;
	       scXtals.clear();
               const std::vector<DetId> & v1 = sc.getHitsByDetId();

               for( std::vector<DetId>::const_iterator idsIt = v1.begin(); idsIt != v1.end(); ++idsIt) {
		    yy++;
                    
                    if((*idsIt).subdetId()!=1 || (*idsIt).det()!=3) continue;
		    
	   	    if((barrelHitsCollection->find(*idsIt))->energy() > currEnergy) {
	   		currEnergy=(barrelHitsCollection->find(*idsIt))->energy();
	   		maxHit=*idsIt;
	   	    }
 		   
		    miniEBRecHitCollection->push_back(*(barrelHitsCollection->find(*idsIt)));
                    scXtals.push_back(*idsIt);
                }
 

    
	       for (unsigned int icry=0;icry<25;icry++)
	         {
 		       unsigned int row = icry / 5;
 		       unsigned int column= icry %5;
		       int curr_eta=maxHit.ieta()+column-2;
		       int curr_phi=maxHit.iphi()+row-2;
		       
                       if (curr_eta * maxHit.ieta() <= 0) {if (maxHit.ieta() > 0) curr_eta--; else curr_eta++; }  // JUMP over 0
                       if (curr_phi < 1) curr_phi += 360;
                       if (curr_phi > 360) curr_phi -= 360;

		       if(curr_eta > 85) continue;
		       if(curr_eta < -85) continue;
		       if(curr_phi > 360) continue;
		       if(curr_phi < 0) continue;

			         EBDetId det = EBDetId(curr_eta,curr_phi,EBDetId::ETAPHIMODE);

 			         std::vector<EBDetId>::const_iterator usedIds;
				 
				 bool HitAlreadyUsed=false;
 				    for(usedIds=scXtals.begin(); usedIds!=scXtals.end(); usedIds++)
				    {
 				       if(*usedIds==det){
                                           HitAlreadyUsed=true;
					}
				    }
				  
				 if(!HitAlreadyUsed){
				      miniEBRecHitCollection->push_back(*(barrelHitsCollection->find(det)));
				  }
				       
	         }
	  

         miniSuperClusterCollection->push_back(sc);
         miniSiStripElectronCollection->push_back(*siStripEleIt);
       }
    }


  //Put selected information in the event
   iEvent.put( miniEBRecHitCollection,alcaBarrelHitsCollection_ );
   iEvent.put( miniSuperClusterCollection,alcaHybridSuperClusterCollection_ );
   iEvent.put( miniSiStripElectronCollection,alcaSiStripElectronCollection_ );
 
}
