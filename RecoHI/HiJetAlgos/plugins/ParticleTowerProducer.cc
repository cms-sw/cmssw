// -*- C++ -*-
//
// Package:    ParticleTowerProducer
// Class:      ParticleTowerProducer
// 
/**\class ParticleTowerProducer ParticleTowerProducer.cc RecoHI/ParticleTowerProducer/src/ParticleTowerProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz,32 4-A08,+41227673039,
//         Created:  Thu Jan 20 19:53:58 CET 2011
// $Id: ParticleTowerProducer.cc,v 1.5 2011/03/15 13:26:21 mnguyen Exp $
//
//


// system include files
#include <memory>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"



#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "TMath.h"
#include "TRandom.h"


 
//
// class declaration
//

class iAngle{
public:
   int ieta;
   int iphi;

   int mag() const { return ieta*1000+iphi;}

   bool operator < (const iAngle& b) {return mag() < b.mag();}
   bool operator > (const iAngle& b) {return mag() > b.mag();}

   friend bool operator < (const iAngle& a, const iAngle& b) {return a.mag() < b.mag();}
   friend bool operator > (const iAngle& a, const iAngle& b) {return a.mag() > b.mag();}


};


class ParticleTowerProducer : public edm::EDProducer {
   public:
      explicit ParticleTowerProducer(const edm::ParameterSet&);
      ~ParticleTowerProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
   void resetTowers(edm::Event& iEvent,const edm::EventSetup& iSetup);
  DetId getNearestTower(const reco::Candidate & in) const;

      // ----------member data ---------------------------

   edm::InputTag src_;

  std::map<DetId,double> towers_;

  //unsigned int nEta_;
  //unsigned int nPhi_;

   double PI;
   TRandom* random_;

  CaloGeometry const *  geo_;                       // geometry
  
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
ParticleTowerProducer::ParticleTowerProducer(const edm::ParameterSet& iConfig):
   geo_(0)
{
   //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
*/
   //now do what ever other initialization is needed
  
  src_ = iConfig.getParameter<edm::InputTag>("src");
  
  produces<CaloTowerCollection>();

   random_ = new TRandom();
   PI = TMath::Pi();



}


ParticleTowerProducer::~ParticleTowerProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ParticleTowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   if(!geo_){
      edm::ESHandle<CaloGeometry> pG;
      iSetup.get<CaloGeometryRecord>().get(pG);
      geo_ = pG.product();
   }


   resetTowers(iEvent, iSetup);


 
   edm::Handle<edm::View<reco::Candidate> > inputsHandle;
   iEvent.getByLabel(src_, inputsHandle);

   for(unsigned int i = 0; i < inputsHandle->size(); ++i){
      const reco::Candidate & particle = (*inputsHandle)[i];

      // put a cutoff if you want
      //if(particle.et() < 0.3) continue;      

      double eta  = particle.eta();
      double phi  = particle.phi();

      HcalDetId hid = getNearestTower(particle);

      towers_[hid] += particle.et();      
   }

   
   std::auto_ptr<CaloTowerCollection> prod(new CaloTowerCollection());

   for ( std::map< DetId, double >::const_iterator iter = towers_.begin();
	 iter != towers_.end(); ++iter ){
     
     CaloTowerDetId newTowerId(iter->first.rawId());
     double et = iter->second;

     if(et>0){

       GlobalPoint pos =geo_->getGeometry(newTowerId)->getPosition();
       
       // currently sets et =  pt, mass to zero
       // pt, eta , phi, mass
       reco::Particle::PolarLorentzVector p4(et,pos.eta(),pos.phi(),0.);
       
       CaloTower newTower(newTowerId,et,0,0,0,0,p4,pos,pos);
       prod->push_back(newTower);     
     }
   }

   
   //For reference, Calo Tower Constructors

   /*
   CaloTower(const CaloTowerDetId& id, 
             double emE, double hadE, double outerE,
             int ecal_tp, int hcal_tp,
             const PolarLorentzVector p4,
       GlobalPoint emPosition, GlobalPoint hadPosition);
 
   CaloTower(const CaloTowerDetId& id, 
             double emE, double hadE, double outerE,
             int ecal_tp, int hcal_tp,
             const LorentzVector p4,
       GlobalPoint emPosition, GlobalPoint hadPosition);
   */


   iEvent.put(prod);


}

// ------------ method called once each job just before starting event loop  ------------
void 
ParticleTowerProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ParticleTowerProducer::endJob() {
}


void ParticleTowerProducer::resetTowers(edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  
  std::vector<DetId> alldid =  geo_->getValidDetIds();

  for(std::vector<DetId>::const_iterator did=alldid.begin(); did != alldid.end(); did++){
    if( (*did).det() == DetId::Hcal ){
       HcalDetId hid = HcalDetId(*did);
       if( hid.depth() == 1 ) {
	  towers_[(*did)] = 0.;
       }
       
    }
  }

}




DetId ParticleTowerProducer::getNearestTower(const reco::Candidate & in) const {

  double eta = in.eta();
  double phi  = in.phi();

  double minDeltaR = 9999;

  std::vector<DetId> alldid =  geo_->getValidDetIds();

  DetId returnId;
  
  //int nclosetowers=0;

  for(std::vector<DetId>::const_iterator did=alldid.begin(); did != alldid.end(); did++){
    if( (*did).det() == DetId::Hcal ){

      HcalDetId hid(*did);
      
      // which layer is irrelevant for an eta-phi map, no?

      if( hid.depth() != 1 ) continue;

      GlobalPoint pos =geo_->getGeometry(hid)->getPosition();
      
      double hcalEta = pos.eta();
      double hcalPhi = pos.phi();

      //std::cout<<" ieta "<<(hid).ieta()<<" iphi "<<(hid).iphi()<<" hcalEta "<<hcalEta<<" hcalPhi "<<hcalPhi<<std::endl;

      double deltaR = reco::deltaR(eta,phi,hcalEta,hcalPhi);
      
      // need to factor in the size of the tower
      double towersize = 0.087;
     
      int ieta  = (hid).ieta();
      
      if(abs(ieta)>21){
	if(abs(ieta)>29) towersize=0.175;
	else{
	  if(abs(ieta)==22) towersize=0.1;
	  else if(abs(ieta)==23) towersize=0.113;
	  else if(abs(ieta)==24) towersize=0.129;
	  else if(abs(ieta)==25) towersize=0.16;
	  else if(abs(ieta)==26) towersize=0.168;
	  else if(abs(ieta)==27) towersize=0.15;
	  else if(abs(ieta)==28) towersize=0.218;
	  else if(abs(ieta)==29) towersize=0.132;
	}
      }

      deltaR /= towersize;
      //if(deltaR<1/3.) nclosetowers++;

      if(deltaR<minDeltaR){
	 returnId = DetId(*did);
	 minDeltaR = deltaR;
      }
      
      //if(abs(eta-hcalEta)<towersize/2. && abs(phi-hcalPhi)<towersize/2.) break;
    }
  }
  //if(nclosetowers>1)std::cout<<"eta "<<eta<<" phi "<<phi<<" minDeltaR "<<minDeltaR<<" nclosetowers "<<nclosetowers<<std::endl;
  return returnId;


}
    
    //define this as a plug-in
DEFINE_FWK_MODULE(ParticleTowerProducer);
