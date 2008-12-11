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
#include "DataFormats/EgammaReco/interface/BasicCluster.h"


AlCaElectronsProducer::AlCaElectronsProducer(const edm::ParameterSet& iConfig) 
{

  ebRecHitsLabel_ = iConfig.getParameter< edm::InputTag > ("ebRecHitsLabel");
  eeRecHitsLabel_ = iConfig.getParameter< edm::InputTag > ("eeRecHitsLabel");
  electronLabel_ = iConfig.getParameter< edm::InputTag > ("electronLabel");

  alcaBarrelHitsCollection_ = iConfig.getParameter<std::string>("alcaBarrelHitCollection");
  alcaEndcapHitsCollection_ = iConfig.getParameter<std::string>("alcaEndcapHitCollection");
  
  etaSize_ = iConfig.getParameter<int> ("etaSize");
  phiSize_ = iConfig.getParameter<int> ("phiSize");
  if ( phiSize_ % 2 == 0 ||  etaSize_ % 2 == 0)
    edm::LogError("AlCaElectronsProducerError") << "Size of eta/phi should be odd numbers";
 
   //register your products
  produces< EBRecHitCollection > (alcaBarrelHitsCollection_) ;
  produces< EERecHitCollection > (alcaEndcapHitsCollection_) ;
  
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

  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > miniEBRecHitCollection (new EBRecHitCollection) ;
  std::auto_ptr< EERecHitCollection > miniEERecHitCollection (new EERecHitCollection) ;

//  loop on SiStrip Electrons
  
  reco::GsfElectronCollection::const_iterator eleIt;
  int ii=0;
  
  for (eleIt=electronCollection->begin(); eleIt!=electronCollection->end(); eleIt++) {
    //PG barrel
    if (fabs(eleIt->eta()) <= 1.479) 
      {
    
    ii++;         
    const reco::SuperCluster& sc = *(eleIt->superCluster()) ;
    
    int yy = 0;
    double currEnergy = 0.;
    EBDetId maxHit(0);
    vector<EBDetId> scXtals;
    scXtals.clear();
    const std::vector<DetId> & v1 = sc.getHitsByDetId();
    
    for(std::vector<DetId>::const_iterator idsIt = v1.begin(); 
        idsIt != v1.end(); ++idsIt) 
      {
        yy++;
        
        //PG discard hits not belonging to the Ecal Barrel
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
        {edm::LogWarning ("shape") << "In Barrel DetId not built Eta "<<curr_eta<<" Phi "<<curr_phi ;}
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
    const std::vector<DetId> & v1 = sc.getHitsByDetId();
    
    for(std::vector<DetId>::const_iterator idsIt = v1.begin(); 
        idsIt != v1.end(); ++idsIt) 
      {
        yy++;
        
        //PG discard hits belonging to the Ecal Barrel
        if((*idsIt).subdetId()!=EcalEndcap || 
           (*idsIt).det()!= DetId::Ecal) continue;
        
        if((endcapHitsCollection->find(*idsIt))->energy() > currEnergy) 
          {
            currEnergy=(endcapHitsCollection->find(*idsIt))->energy();
            maxHit=*idsIt;
          }
        miniEERecHitCollection->push_back (
            *(endcapHitsCollection->find (*idsIt))
          );
        scXtals.push_back (*idsIt) ; 
    }  

    int side = phiSize_ ;
    if (phiSize_ < etaSize_) side = etaSize_ ;
    int iz = 1 ;
    if (eleIt->eta () < 0)  iz = -1 ;
    if (!maxHit.null())
      //PG loop over the local array of xtals
      for (unsigned int icry = 0 ; icry < side*side ; icry++)
        {          
          unsigned int row = icry / side ;
          unsigned int column = icry % side ;
          int curr_eta = maxHit.ix () + column - (side/2);
          int curr_phi = maxHit.iy () + row - (side/2);
          if (   curr_eta <= 0 || curr_phi <= 0
              || curr_eta > 100 || curr_phi > 100 ) continue ;
          
          
          try
            {
              EEDetId det = EEDetId (curr_eta,curr_phi,iz,EEDetId::XYMODE) ; 
              if (find (scXtals.begin (), scXtals.end (), det) != scXtals.end ())
                if (endcapHitsCollection->find (det) != endcapHitsCollection->end ())
                  miniEERecHitCollection->push_back (*(endcapHitsCollection->find (det))) ;
            }
          catch (...)
            {
              edm::LogWarning ("shape") << "DetId (" 
                                        << curr_eta << "," << curr_phi 
                                        << ") not built" ;
//PG              m_failMap->Fill (curr_eta,curr_phi) ;
            }
        }
      } //PG endcap
      
  } //PG loop on Si strip electrons
    
  //Put selected information in the event
  iEvent.put( miniEBRecHitCollection,alcaBarrelHitsCollection_ );
  iEvent.put( miniEERecHitCollection,alcaEndcapHitsCollection_ );     
}
