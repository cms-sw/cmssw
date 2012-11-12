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
// $Id: ParticleTowerProducer.cc,v 1.9 2012/02/14 08:43:00 mnguyen Exp $
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

#include "RecoHI/HiJetAlgos/plugins/ParticleTowerProducer.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "TMath.h"
#include "TRandom.h"




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
  src_ = iConfig.getParameter<edm::InputTag>("src");
  useHF_ = iConfig.getUntrackedParameter<bool>("useHF");
  
  produces<CaloTowerCollection>();
  
  //now do what ever other initialization is needed
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


   edm::Handle<reco::PFCandidateCollection> inputsHandle;
   iEvent.getByLabel(src_, inputsHandle);
   
   for(reco::PFCandidateCollection::const_iterator ci  = inputsHandle->begin(); ci!=inputsHandle->end(); ++ci)  {

    const reco::PFCandidate& particle = *ci;
    
    // put a cutoff if you want
    //if(particle.et() < 0.3) continue;      
    
    double eta = particle.eta();

    if(!useHF_ && fabs(eta) > 3. ) continue;

    int ieta = eta2ieta(eta);
    if(eta<0) ieta  *= -1;

    int iphi = phi2iphi(particle.phi(),ieta);
       
    HcalSubdetector sd;
    if(abs(ieta)<=16) sd = HcalBarrel;
    else if(fabs(eta)< 3.) sd = HcalEndcap;  // Use the endcap until eta =3
    else sd = HcalForward;
    
    HcalDetId hid = HcalDetId(sd,ieta,iphi,1); // assume depth=1

    // check against the old method (boundaries slightly shifted in the endcaps
    /*
    HcalDetId hid_orig = getNearestTower(particle);
    
    if(hid != hid_orig)
      {	
	if(abs((hid).ieta()-(hid_orig).ieta())>1 || abs((hid).iphi()-(hid_orig).iphi()) >1 ){

	  int phi_diff = 1;
	  if(abs((hid).ieta())>39 || abs((hid_orig).ieta())>39) phi_diff = 4;
	  else if(abs((hid).ieta())>20 || abs((hid_orig).ieta())>20) phi_diff = 2;

	  if(abs((hid).ieta()-(hid_orig).ieta()) > 1 || abs((hid).iphi()-(hid_orig).iphi()) > phi_diff ){
	      std::cout<<" eta "<<eta<<std::endl;
	      std::cout<<" hid "<<hid<<std::endl;
	      std::cout<<" old method "<<hid_orig<<std::endl;
	  }
	}
      }
    */
    towers_[hid] += particle.et();      
   }
   
   
   std::auto_ptr<CaloTowerCollection> prod(new CaloTowerCollection());

   for ( std::map< DetId, double >::const_iterator iter = towers_.begin();
	 iter != towers_.end(); ++iter ){
     
     CaloTowerDetId newTowerId(iter->first.rawId());
     double et = iter->second;

     if(et>0){

       GlobalPoint pos =geo_->getGeometry(newTowerId)->getPosition();
       
       if(!useHF_ && fabs(pos.eta()) > 3. ) continue;

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
  // tower edges from fast sim, used starting at index 30 for the HF
  const double etatow[42] = {0.000, 0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.870, 0.957, 1.044, 1.131, 1.218, 1.305, 1.392, 1.479, 1.566, 1.653, 1.740, 1.830, 1.930, 2.043, 2.172, 2.322, 2.500, 2.650, 2.853, 3.000, 3.139, 3.314, 3.489, 3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889, 5.191};
  
  // tower centers derived directly from by looping over HCAL towers and printing out the values, used up until eta = 3 where the HF/Endcap interface becomes problematic
  const double etacent[40] = {
    0.0435, // 1
    0.1305, // 2
    0.2175, // 3
    0.3045, // 4
    0.3915, // 5
    0.4785, // 6
    0.5655, // 7
    0.6525, // 8
    0.7395, // 9
    0.8265, // 10
    0.9135, // 11
    1.0005, // 12
    1.0875, // 13
    1.1745, // 14
    1.2615, // 15
    1.3485, // 16
    1.4355, // 17
    1.5225, // 18
    1.6095, // 19
    1.6965, // 20
    1.785 , // 21
    1.88  , // 22
    1.9865, // 23
    2.1075, // 24
    2.247 , // 25
    2.411 , // 26
    2.575 , // 27
    2.759 , // 28
    2.934 , // 29
    2.90138,// 29
    3.04448,// 30 
    3.21932,// 31
    3.39462,// 32
    3.56966,// 33 
    3.74485,// 34 
    3.91956,// 35 
    4.09497,// 36 
    4.27007,// 37 
    4.44417,// 38 
    4.62046 // 39 
      };

  // Use the real towers centrers for the barrel and endcap up to eta=3
  etaedge[0] = 0.;
  for(int i=1;i<30;i++){
    etaedge[i] = (etacent[i-1]-etaedge[i-1])*2.0 + etaedge[i-1];
    //std::cout<<" i "<<i<<" etaedge "<<etaedge[i]<<std::endl;  
  }
  // beyond eta = 3 just use the fast sim values
  if(useHF_){
    for(int i=30;i<42;i++){
      etaedge[i]=etatow[i];
      //std::cout<<" i "<<i<<" etaedge "<<etaedge[i]<<std::endl;  
    }
  }
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
	 
	 if(!useHF_){
	   GlobalPoint pos =geo_->getGeometry(hid)->getPosition();	 
	   //if((hid).iphi()==1)std::cout<<" ieta "<<(hid).ieta()<<" eta "<<pos.eta()<<" iphi "<<(hid).iphi()<<" phi "<<pos.phi()<<std::endl;
	   if(fabs(pos.eta())>3.) continue;
	 }
	  towers_[(*did)] = 0.;
       }
       
    }
  }




  //print out tower eta boundaries
  /*
  int current_ieta=0;
  float testphi =0.;
  for(double testeta=1.7; testeta<=5.4;testeta+=0.00001){
    HcalDetId hid = getNearestTower(testeta,testphi);
    if((hid).ieta()!=current_ieta) {
      std::cout<<" new ieta "<<current_ieta<<" testeta "<<testeta<<std::endl;
      current_ieta = (hid).ieta();
    }
  }
  */

}




DetId ParticleTowerProducer::getNearestTower(const reco::PFCandidate & in) const {

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

      //std::cout<<" transverse distance = "<<sqrt(pos.x()*pos.x()+pos.y()*pos.y())<<std::endl;

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


DetId ParticleTowerProducer::getNearestTower(double eta, double phi) const {

  // get closest tower by delta R matching
  // Now obsolute, instead use faster premade eta-phi grid

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

      //std::cout<<" transverse distance = "<<sqrt(pos.x()*pos.x()+pos.y()*pos.y())<<std::endl;

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




// Taken from FastSimulation/CalorimeterProperties/src/HCALProperties.cc
// Note this returns an abs(ieta)
int ParticleTowerProducer::eta2ieta(double eta) const {
  // binary search in the array of towers eta edges
  int size = 42;
  if(!useHF_) size = 30;

  double x = fabs(eta);
  int curr = size / 2;
  int step = size / 4;
  int iter;
  int prevdir = 0; 
  int actudir = 0; 

  for (iter = 0; iter < size ; iter++) {

    if( curr >= size || curr < 1 )
      std::cout <<  " ParticleTowerProducer::eta2ieta - wrong current index = "
		<< curr << " !!!" << std::endl;

    if ((x <= etaedge[curr]) && (x > etaedge[curr-1])) break;
    prevdir = actudir;
    if(x > etaedge[curr]) {actudir =  1;}
    else {actudir = -1;}
    if(prevdir * actudir < 0) { if(step > 1) step /= 2;}
    curr += actudir * step;
    if(curr > size) curr = size;
    else { if(curr < 1) {curr = 1;}}

    /*
    std::cout << " HCALProperties::eta2ieta  end of iter." << iter 
          << " curr, etaedge[curr-1], etaedge[curr] = "
	        << curr << " " << etaedge[curr-1] << " " << etaedge[curr] << std::endl;
    */
    
  }

  /*
  std::cout << " HCALProperties::eta2ieta  for input x = " << x 
      << "  found index = " << curr-1
          << std::endl;
  */
  
  return curr;
}

int ParticleTowerProducer::phi2iphi(double phi, int ieta) const {
  
  if(phi<0) phi += 2.*PI;
  else if(phi> 2.*PI) phi -= 2.*PI;
  
  int iphi = (int) TMath::Ceil(phi/2.0/PI*72.);
  // take into account larger granularity in endcap (x2) and at the end of the HF (x4)
  if(abs(ieta)>20){
    if(abs(ieta)<40) iphi -= (iphi+1)%2;
    else {
      iphi -= (iphi+1)%4;
      if(iphi==-1) iphi=71;
      }
  }
  
  return iphi;

}

    //define this as a plug-in
DEFINE_FWK_MODULE(ParticleTowerProducer);
