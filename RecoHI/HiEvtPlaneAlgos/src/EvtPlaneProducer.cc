// -*- C++ -*-
//
// Package:    EvtPlaneProducer
// Class:      EvtPlaneProducer
// 
/**\class EvtPlaneProducer EvtPlaneProducer.cc RecoHI/EvtPlaneProducer/src/EvtPlaneProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sergey Petrushanko
//         Created:  Fri Jul 11 10:05:00 2008
// $Id: EvtPlaneProducer.cc,v 1.8 2010/05/17 11:05:24 yilmaz Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <time.h>
#include "TMath.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <cstdlib>
using std::rand;

using namespace std;

//
// class decleration
//

class EvtPlaneProducer : public edm::EDProducer {
   public:
      explicit EvtPlaneProducer(const edm::ParameterSet&);
      ~EvtPlaneProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  bool useECAL_;
  bool useHCAL_;
  bool useTrackMidEta_;
  bool useTrackPosEta_;
  bool useTrackNegEta_;
  bool useTrackEta_;
  bool useTrackOdd_;
  bool useTrackEven_;
  bool useTrackPosCh_;
  bool useTrackNegCh_;
  bool useTrackPosEtaGap_;
  bool useTrackNegEtaGap_;


  edm::InputTag mcSrc_;
  edm::InputTag trackSrc_;
  edm::InputTag towerSrc_;


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
EvtPlaneProducer::EvtPlaneProducer(const edm::ParameterSet& iConfig)
{

int k=100;
int iseed=k;

  srand(iseed);

   //register your products
  
  useECAL_ = iConfig.getUntrackedParameter<bool>("useECAL",true);
  useHCAL_ = iConfig.getUntrackedParameter<bool>("useHCAL",true);
  useTrackMidEta_ = iConfig.getUntrackedParameter<bool>("useTrackMidEta",true);
  useTrackPosEta_ = iConfig.getUntrackedParameter<bool>("useTrackPosEta",true);
  useTrackNegEta_ = iConfig.getUntrackedParameter<bool>("useTrackNegEta",true);
  useTrackEta_ = iConfig.getUntrackedParameter<bool>("useTrackEta",true);
  useTrackOdd_ = iConfig.getUntrackedParameter<bool>("useTrackOdd",true);
  useTrackEven_ = iConfig.getUntrackedParameter<bool>("useTrackEven",true);
  useTrackPosCh_ = iConfig.getUntrackedParameter<bool>("useTrackPosCh",true);
  useTrackNegCh_ = iConfig.getUntrackedParameter<bool>("useTrackNegCh",true);
  useTrackPosEtaGap_ = iConfig.getUntrackedParameter<bool>("useTrackPosEtaGap",true);
  useTrackNegEtaGap_ = iConfig.getUntrackedParameter<bool>("useTrackNegEtaGap",true);


  mcSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("mc",edm::InputTag("generator"));
  trackSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("tracks",edm::InputTag("hiSelectedTracks"));
  towerSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("towers",edm::InputTag("towerMaker"));

  produces<reco::EvtPlaneCollection>("recoLevel");

}


EvtPlaneProducer::~EvtPlaneProducer()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EvtPlaneProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  using namespace HepMC;

  //calorimetry part

      double ugol[24], ugol2[24];
      double etugol[24], etugol2[24];
      double tower_eta, tower_phi;
      double tower_energy, tower_energy_e, tower_energy_h;
      double tower_energyet, tower_energyet_e, tower_energyet_h;
      double s1t, s2t, s1e, s2e, s1h, s2h, s1t1, s2t1;
      double ets1t, ets2t, ets1e, ets2e, ets1h, ets2h, ets1t1, ets2t1;
//      double pi = 3.14159;
      double s1[24], s2[24];
      double ets1[24], ets2[24];
      
//      double TEnergy[144], TPhi[144];
//      int numb;	

//      double planeA     =  0.;

//       cout << endl << "  Start of the event plane determination." << endl;

       for(int j=0;j<24;j++) {
        s1[j]  = 0.;
        s2[j]  = 0.;
        ets1[j]  = 0.;
        ets2[j]  = 0.;
       }
      
//       for(int l=0;l<144;l++) {
//        TEnergy[l]  = 0.;
//        TPhi[l]  = 0.;
//       }

      Handle<CaloTowerCollection> calotower;
      iEvent.getByLabel(towerSrc_,calotower);
      
      if(!calotower.isValid()){
        cout << "Error! Can't get calotower product!" << endl;
       return ;
      }

	for (CaloTowerCollection::const_iterator j = calotower->begin();j !=calotower->end(); j++) {

//        cout << *j << std::endl;
//        cout << "ENERGY HAD " << j->hadEnergy()<< " ENERGY EM " <<j->emEnergy() 
//	  << " ETA " <<j->eta() << " PHI " <<j->phi() << std::endl;
    
        tower_eta        = j->eta();
        tower_phi        = j->phi();
	tower_energy_e   = j->emEnergy();
	tower_energy_h   = j->hadEnergy();
        tower_energy     = tower_energy_e + tower_energy_h;
	tower_energyet_e   = j->emEt();
	tower_energyet_h   = j->hadEt();
        tower_energyet     = tower_energyet_e + tower_energyet_h;
	
	s1t = tower_energy*sin(2.*tower_phi);
	s2t = tower_energy*cos(2.*tower_phi);
        s1e = tower_energy_e*sin(2.*tower_phi);
	s2e = tower_energy_e*cos(2.*tower_phi);
	s1h = tower_energy_h*sin(2.*tower_phi);
	s2h = tower_energy_h*cos(2.*tower_phi);

	s1t1 = tower_energy*sin(tower_phi);
	s2t1 = tower_energy*cos(tower_phi);

	ets1t = tower_energyet*sin(2.*tower_phi);
	ets2t = tower_energyet*cos(2.*tower_phi);
        ets1e = tower_energyet_e*sin(2.*tower_phi);
	ets2e = tower_energyet_e*cos(2.*tower_phi);
	ets1h = tower_energyet_h*sin(2.*tower_phi);
	ets2h = tower_energyet_h*cos(2.*tower_phi);

	ets1t1 = tower_energyet*sin(tower_phi);
	ets2t1 = tower_energyet*cos(tower_phi);

// ENERGY
	  
	   s1[12] += s1t;
	   s2[12] += s2t;
	  
	   ets1[12] += ets1t;
	   ets2[12] += ets2t;

	  if (tower_eta>0.25)
	  {
  	   s1[19] +=  s1t;
	   s2[19] +=  s2t;	  
	  }
	  
	  if (tower_eta<-0.25)
	  {
  	   s1[20] +=  s1t;
	   s2[20] +=  s2t;	  
	  }	

	 if (fabs(tower_eta)>3.){ 
	  if (tower_eta>3.) {
	   s1[9] += s1t;
	   s2[9] += s2t;
	   s1[21] += s1t1;
	   s2[21] += s2t1;
	   
	  }
	  if (tower_eta<-3.) {
	   s1[10] += s1t;
	   s2[10] += s2t;
	   s1[22] += s1t1;
	   s2[22] += s2t1;
	  }
	   s1[11] += s1t;
	   s2[11] += s2t;
	   s1[23] += s1t1;
	   s2[23] += s2t1;
	 }

	 if (fabs(tower_eta)<3.){

  	  s1[0] +=  s1t;
	  s2[0] +=  s2t;
  	  s1[3] +=  s1h;
	  s2[3] +=  s2h;
  	  s1[6] +=  s1e;
	  s2[6] +=  s2e;
	  
	  if (tower_eta>0.25)
	  {
  	   s1[13] +=  s1e;
	   s2[13] +=  s2e;
  	   s1[15] +=  s1h;
	   s2[15] +=  s2h;
  	   s1[17] +=  s1t;
	   s2[17] +=  s2t;	  
	  }
	  
	  if (tower_eta<-0.25)
	  {
  	   s1[14] +=  s1e;
	   s2[14] +=  s2e;
  	   s1[16] +=  s1h;
	   s2[16] +=  s2h;
  	   s1[18] +=  s1t;
	   s2[18] +=  s2t;	  
	  }
	  	  
	 if (fabs(tower_eta)>1.5) {
  	  s1[2] +=  s1t;
	  s2[2] +=  s2t;
  	  s1[5] +=  s1h;
	  s2[5] +=  s2h;
  	  s1[8] +=  s1e;
	  s2[8] +=  s2e;
	 }
	 }
	 
	 if (fabs(tower_eta)<1.5){
  	  s1[1] +=  s1t;
	  s2[1] +=  s2t;
  	  s1[4] +=  s1h;
	  s2[4] +=  s2h;
  	  s1[7] +=  s1e;
	  s2[7] +=  s2e;
	 }
	 
	  if (tower_eta>0.25)
	  {
  	   ets1[19] +=  ets1t;
	   ets2[19] +=  ets2t;	  
	  }
	  
	  if (tower_eta<-0.25)
	  {
  	   ets1[20] +=  ets1t;
	   ets2[20] +=  ets2t;	  
	  }	

	 if (fabs(tower_eta)>3.){ 
	  if (tower_eta>3.) {
	   ets1[9] += ets1t;
	   ets2[9] += ets2t;
	   ets1[21] += ets1t1;
	   ets2[21] += ets2t1;
	  }
	  if (tower_eta<-3.) {
	   ets1[10] += ets1t;
	   ets2[10] += ets2t;
	   ets1[22] += ets1t1;
	   ets2[22] += ets2t1;
	  }
	   ets1[11] += ets1t;
	   ets2[11] += ets2t;
	   ets1[23] += ets1t1;
	   ets2[23] += ets2t1;
	 }

	 if (fabs(tower_eta)<3.){

  	  ets1[0] +=  ets1t;
	  ets2[0] +=  ets2t;
  	  ets1[3] +=  ets1h;
	  ets2[3] +=  ets2h;
  	  ets1[6] +=  ets1e;
	  ets2[6] +=  ets2e;
	  
	  if (tower_eta>0.25)
	  {
  	   ets1[13] +=  ets1e;
	   ets2[13] +=  ets2e;
  	   ets1[15] +=  ets1h;
	   ets2[15] +=  ets2h;
  	   ets1[17] +=  ets1t;
	   ets2[17] +=  ets2t;	  
	  }
	  
	  if (tower_eta<-0.25)
	  {
  	   ets1[14] +=  ets1e;
	   ets2[14] +=  ets2e;
  	   ets1[16] +=  ets1h;
	   ets2[16] +=  ets2h;
  	   ets1[18] +=  ets1t;
	   ets2[18] +=  ets2t;	  
	  }
	  	  
	 if (fabs(tower_eta)>1.5) {
  	  ets1[2] +=  ets1t;
	  ets2[2] +=  ets2t;
  	  ets1[5] +=  ets1h;
	  ets2[5] +=  ets2h;
  	  ets1[8] +=  ets1e;
	  ets2[8] +=  ets2e;
	 }
	 }
	 
	 if (fabs(tower_eta)<1.5){
  	  ets1[1] +=  ets1t;
	  ets2[1] +=  ets2t;
  	  ets1[4] +=  ets1h;
	  ets2[4] +=  ets2h;
  	  ets1[7] +=  ets1e;
	  ets2[7] +=  ets2e;
	 }	 
	 
	}
	
      for(int j1=0;j1<24;j1++) {
 
       if (s2[j1]==0.) {ugol[j1]=0.;}
       else {ugol[j1] = 0.5*atan2(s1[j1],s2[j1]);}
       
       if ((j1==21) || (j1==22) || (j1==23))
      {
       if (s2[j1]==0.) {ugol[j1]=0.;}
       else {ugol[j1] = atan2(s1[j1],s2[j1]);}       
       }
       
       ugol2[j1] = ugol[j1];
       
      }

/*
       cout <<  endl << "   Azimuthal angle of reaction plane (with maximum), energy" << endl
       << "HCAL+ECAL (b+e)   " << ugol2[0] << endl
       << "HCAL+ECAL (b)     " << ugol2[1] << endl
       << "HCAL+ECAL (e)     " << ugol2[2] << endl
       << "HCAL      (b+e)   " << ugol2[3] << endl
       << "HCAL      (b)     " << ugol2[4] << endl
       << "HCAL      (e)     " << ugol2[5] << endl
       << "ECAL      (b+e)   " << ugol2[6] << endl
       << "ECAL      (b)     " << ugol2[7] << endl
       << "ECAL      (e)     " << ugol2[8] << endl  
       << "HF       (plus)   " << ugol2[9] << endl
       << "HF       (minus)  " << ugol2[10] << endl
       << "HF       (p+m)    " << ugol2[11] << endl
       << "HCAL+ECAL+HF      " << ugol2[12] << endl
       << "ECAL      (plus)  " << ugol2[13] << endl
       << "ECAL      (minus) " << ugol2[14] << endl
       << "HCAL      (plus)  " << ugol2[15] << endl
       << "HCAL      (minus) " << ugol2[16] << endl
       << "HCAL+ECAL (plus)  " << ugol2[17] << endl
       << "HCAL+ECAL (minus) " << ugol2[18] << endl
       << "HCAL+ECAL+HF (p)  " << ugol2[19] << endl
       << "HCAL+ECAL+HF (m)  " << ugol2[20] << endl
       << endl;
*/


// ET ENERGY

	
      for(int j1=0;j1<24;j1++) {
 
       if (ets2[j1]==0.) {etugol[j1]=0.;}
       else {etugol[j1] = 0.5*atan2(ets1[j1],ets2[j1]);}
       
       if ((j1==21) || (j1==22) || (j1==23))
      {
       if (ets2[j1]==0.) {etugol[j1]=0.;}
       else {etugol[j1] = atan2(ets1[j1],ets2[j1]);}       
       }
              
       etugol2[j1] = etugol[j1];
       
      }

/*
       cout <<  endl << "   Azimuthal angle of reaction plane (with maximum), t energy" << endl
       << "HCAL+ECAL (b+e)   " << etugol2[0] << endl
       << "HCAL+ECAL (b)     " << etugol2[1] << endl
       << "HCAL+ECAL (e)     " << etugol2[2] << endl
       << "HCAL      (b+e)   " << etugol2[3] << endl
       << "HCAL      (b)     " << etugol2[4] << endl
       << "HCAL      (e)     " << etugol2[5] << endl
       << "ECAL      (b+e)   " << etugol2[6] << endl
       << "ECAL      (b)     " << etugol2[7] << endl
       << "ECAL      (e)     " << etugol2[8] << endl  
       << "HF       (plus)   " << etugol2[9] << endl
       << "HF       (minus)  " << etugol2[10] << endl
       << "HF       (p+m)    " << etugol2[11] << endl
       << "HCAL+ECAL+HF      " << etugol2[12] << endl
       << "ECAL      (plus)  " << etugol2[13] << endl
       << "ECAL      (minus) " << etugol2[14] << endl
       << "HCAL      (plus)  " << etugol2[15] << endl
       << "HCAL      (minus) " << etugol2[16] << endl
       << "HCAL+ECAL (plus)  " << etugol2[17] << endl
       << "HCAL+ECAL (minus) " << etugol2[18] << endl
       << "HCAL+ECAL+HF (p)  " << etugol2[19] << endl
       << "HCAL+ECAL+HF (m)  " << etugol2[20] << endl
       << endl;


   const GenEvent *evt;

   Handle<HepMCProduct> mc;
   iEvent.getByLabel(mcSrc_,mc);
   evt = mc->GetEvent();

   const HeavyIon* hi = evt->heavy_ion();
   double impactP = hi->impact_parameter();
   double phi0 = hi->event_plane_angle();

   cout<<"  Generator Ugol " << phi0 <<  " Impact parameter  " << impactP << endl;
   
*/


//Tracking part
     
      double track_eta;
      double track_phi;
      double track_pt;
      double track_charge;

   double trackPsi_eta_mid;
   double trackPsi_eta_pos;
   double trackPsi_eta_neg;
   double trackPsi_eta_posgap;
   double trackPsi_eta_neggap;
   double trackPsi_odd;
   double trackPsi_even;
   double trackPsi_eta;
   double trackPsi_PosCh;
   double trackPsi_NegCh;


   Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(trackSrc_, tracks);

   // cout << " TRACKS Size " << tracks->size() << endl;

   if(!tracks.isValid()){
     cout << "Error! Can't get selectTracks!" << endl;
     return ;
   }
   double trackSin_eta_mid = 0;
   double trackCos_eta_mid = 0;
   double trackSin_eta_pos = 0;
   double trackCos_eta_pos = 0;
   double trackSin_eta_neg = 0;
   double trackCos_eta_neg = 0;
   double trackSin_eta_posgap = 0;
   double trackCos_eta_posgap = 0;
   double trackSin_eta_neggap = 0;
   double trackCos_eta_neggap = 0;
   double trackSin_eta = 0;
   double trackCos_eta = 0;
   double trackSin_even = 0;
   double trackCos_even = 0;
   double trackSin_odd = 0;
   double trackCos_odd = 0;
   double trackSin_PosCh = 0;
   double trackCos_PosCh = 0;
   double trackSin_NegCh = 0;
   double trackCos_NegCh = 0;
   double face=0; 
   double s=0;
   
   for(reco::TrackCollection::const_iterator j = tracks->begin(); j != tracks->end(); j++){
    
     track_eta = j->eta();
     track_phi = j->phi();
     track_pt = j->pt();
     track_charge = j->charge();
     if ((track_phi<1.0) || (track_phi>1.16)){

     if (fabs(track_eta) < 2.0){
       trackSin_eta+=sin(2*track_phi);
       trackCos_eta+=cos(2*track_phi);
     }
     s = tracks->size();
      
     if((track_charge > 0) && (fabs(track_eta) < 0.75) ){
       trackSin_PosCh+=sin(2*track_phi);
       trackCos_PosCh+=cos(2*track_phi);
     }
     if((track_charge < 0) && (fabs(track_eta) < 0.75)){
       trackSin_NegCh+=sin(2*track_phi);
       trackCos_NegCh+=cos(2*track_phi);
     }
     int rnd = rand();
     face = (double)rnd/(double)RAND_MAX;
     
     if (fabs(track_eta) < 0.75){
     if (face < 0.5){
       trackSin_even+=sin(2*track_phi);
       trackCos_even+=cos(2*track_phi);
     }
     
     else {
       trackSin_odd+=sin(2*track_phi);
       trackCos_odd+=cos(2*track_phi);
     }
     }
     if(fabs(track_eta)<0.75){
       trackSin_eta_mid+=sin(2*track_phi);
       trackCos_eta_mid+=cos(2*track_phi);
     }
   
     if((track_eta >= 0.75) && (track_eta < 2.0)){
       trackSin_eta_pos+=sin(2*track_phi);
       trackCos_eta_pos+=cos(2*track_phi);
     }
     if((track_eta <= -0.75) && (track_eta > -2.0)){
       trackSin_eta_neg+=sin(2*track_phi);
       trackCos_eta_neg+=cos(2*track_phi);
     }
   
     if((track_eta >= 1) && (track_eta < 2.0)){
       trackSin_eta_posgap+=sin(2*track_phi);
       trackCos_eta_posgap+=cos(2*track_phi);
     }
     if((track_eta <= -1) && (track_eta > -2.0)){
       trackSin_eta_neggap+=sin(2*track_phi);
       trackCos_eta_neggap+=cos(2*track_phi);
     }

   }
   }
        trackPsi_eta_mid = 0.5*atan2(trackSin_eta_mid,trackCos_eta_mid);
	trackPsi_eta_pos = 0.5*atan2(trackSin_eta_pos,trackCos_eta_pos);
	trackPsi_eta_neg = 0.5*atan2(trackSin_eta_neg,trackCos_eta_neg);
	trackPsi_eta = 0.5*atan2(trackSin_eta,trackCos_eta);
	trackPsi_odd = 0.5*atan2(trackSin_odd,trackCos_odd);
	trackPsi_even = 0.5*atan2(trackSin_even,trackCos_even);
	trackPsi_PosCh = 0.5*atan2(trackSin_PosCh,trackCos_PosCh);
        trackPsi_NegCh = 0.5*atan2(trackSin_NegCh,trackCos_NegCh);
	trackPsi_eta_posgap = 0.5*atan2(trackSin_eta_posgap,trackCos_eta_posgap);
	trackPsi_eta_neggap = 0.5*atan2(trackSin_eta_neggap,trackCos_eta_neggap);

//  cout << "  "<< trackPsi_eta_mid << " "<< trackPsi_odd << endl;

   std::auto_ptr<EvtPlaneCollection> evtplaneOutput(new EvtPlaneCollection);
    
   EvtPlane ecalPlane(ugol2[6],s1[6],s2[6],"Ecal");
   EvtPlane hcalPlane(ugol2[3],s1[3],s2[3],"Hcal");
   EvtPlane hfPlane(ugol2[11],s1[11],s2[11],"HF");
   EvtPlane caloPlane(ugol2[0],s1[0],s2[0],"Calo");
   EvtPlane calohfPlane(ugol2[12],s1[12],s2[12],"CaloHF");

   EvtPlane ecalpPlane(ugol2[13],s1[13],s2[13],"EcalP");
   EvtPlane ecalmPlane(ugol2[14],s1[14],s2[14],"EcalM");
   EvtPlane hcalpPlane(ugol2[15],s1[15],s2[15],"HcalP");
   EvtPlane hcalmPlane(ugol2[16],s1[16],s2[16],"HcalM");
   EvtPlane hfpPlane(ugol2[9],s1[9],s2[9],"HFp");
   EvtPlane hfmPlane(ugol2[10],s1[10],s2[10],"HFm");
   EvtPlane calopPlane(ugol2[17],s1[17],s2[17],"CaloP");
   EvtPlane calomPlane(ugol2[18],s1[18],s2[18],"CaloM");
   EvtPlane calohfpPlane(ugol2[19],s1[19],s2[19],"CaloHFP");
   EvtPlane calohfmPlane(ugol2[20],s1[20],s2[20],"CaloHFM");
   
   EvtPlane hf1Plane(ugol2[23],s1[23],s2[23],"HF1");
   EvtPlane hfp1Plane(ugol2[21],s1[21],s2[21],"HFp1");
   EvtPlane hfm1Plane(ugol2[22],s1[22],s2[22],"HFm1");

   EvtPlane etecalPlane(etugol2[6],ets1[6],ets2[6],"etEcal");
   EvtPlane ethcalPlane(etugol2[3],ets1[3],ets2[3],"etHcal");
   EvtPlane ethfPlane(etugol2[11],ets1[11],ets2[11],"etHF");
   EvtPlane etcaloPlane(etugol2[0],ets1[0],ets2[0],"etCalo");
   EvtPlane etcalohfPlane(etugol2[12],ets1[12],ets2[12],"etCaloHF");

   EvtPlane etecalpPlane(etugol2[13],ets1[13],ets2[13],"etEcalP");
   EvtPlane etecalmPlane(etugol2[14],ets1[14],ets2[14],"etEcalM");
   EvtPlane ethcalpPlane(etugol2[15],ets1[15],ets2[15],"etHcalP");
   EvtPlane ethcalmPlane(etugol2[16],ets1[16],ets2[16],"etHcalM");
   EvtPlane ethfpPlane(etugol2[9],ets1[9],ets2[9],"etHFp");
   EvtPlane ethfmPlane(etugol2[10],ets1[10],ets2[10],"etHFm");
   EvtPlane etcalopPlane(etugol2[17],ets1[17],ets2[17],"etCaloP");
   EvtPlane etcalomPlane(etugol2[18],ets1[18],ets2[18],"etCaloM");
   EvtPlane etcalohfpPlane(etugol2[19],ets1[19],ets2[19],"etCaloHFP");
   EvtPlane etcalohfmPlane(etugol2[20],ets1[20],ets2[20],"etCaloHFM");

   EvtPlane ethf1Plane(etugol2[23],ets1[23],ets2[23],"etHF1");
   EvtPlane ethfp1Plane(etugol2[21],ets1[21],ets2[21],"etHFp1");
   EvtPlane ethfm1Plane(etugol2[22],ets1[22],ets2[22],"etHFm1");


   EvtPlane EvtPlaneFromTracksMidEta(trackPsi_eta_mid,trackSin_eta_mid,trackCos_eta_mid,"EvtPlaneFromTracksMidEta");
   EvtPlane EvtPlaneFromTracksPosEta(trackPsi_eta_pos,trackSin_eta_pos,trackCos_eta_pos,"EvtPlaneFromTracksPosEta");
   EvtPlane EvtPlaneFromTracksNegEta(trackPsi_eta_neg,trackSin_eta_neg,trackCos_eta_neg,"EvtPlaneFromTracksNegEta");
   EvtPlane EvtPlaneFromTracksEta(trackPsi_eta,trackSin_eta,trackCos_eta,"EvtPlaneFromTracksEta");  
   EvtPlane EvtPlaneFromTracksOdd(trackPsi_odd,trackSin_odd,trackCos_odd,"EvtPlaneFromTracksOdd");
   EvtPlane EvtPlaneFromTracksEven(trackPsi_even,trackSin_even,trackCos_even,"EvtPlaneFromTracksEven");
   EvtPlane EvtPlaneFromTracksPosCh(trackPsi_PosCh,trackSin_PosCh,trackCos_PosCh,"EvtPlaneFromTracksPosCh");
   EvtPlane EvtPlaneFromTracksNegCh(trackPsi_NegCh,trackSin_NegCh,trackCos_NegCh,"EvtPlaneFromTracksNegCh");
   EvtPlane EvtPTracksPosEtaGap(trackPsi_eta_posgap,trackSin_eta_posgap,trackCos_eta_posgap,"EvtPTracksPosEtaGap");
   EvtPlane EvtPTracksNegEtaGap(trackPsi_eta_neggap,trackSin_eta_neggap,trackCos_eta_neggap,"EvtPTracksNegEtaGap");

   if(useTrackMidEta_) evtplaneOutput->push_back(EvtPlaneFromTracksMidEta);
   if(useTrackPosEta_) evtplaneOutput->push_back(EvtPlaneFromTracksPosEta);
   if(useTrackNegEta_) evtplaneOutput->push_back(EvtPlaneFromTracksNegEta);
   if(useTrackEta_) evtplaneOutput->push_back(EvtPlaneFromTracksEta);
   if(useTrackOdd_) evtplaneOutput->push_back(EvtPlaneFromTracksOdd);
   if(useTrackEven_) evtplaneOutput->push_back(EvtPlaneFromTracksEven);
   if(useTrackPosCh_) evtplaneOutput->push_back(EvtPlaneFromTracksPosCh);
   if(useTrackNegCh_) evtplaneOutput->push_back(EvtPlaneFromTracksNegCh);
   if(useTrackPosEtaGap_) evtplaneOutput->push_back(EvtPTracksPosEtaGap);
   if(useTrackNegEtaGap_) evtplaneOutput->push_back(EvtPTracksNegEtaGap);

 
   if(useECAL_) 
    {
     evtplaneOutput->push_back(ecalPlane);
     evtplaneOutput->push_back(ecalpPlane);
     evtplaneOutput->push_back(ecalmPlane);
     evtplaneOutput->push_back(etecalPlane);
     evtplaneOutput->push_back(etecalpPlane);
     evtplaneOutput->push_back(etecalmPlane);
    }
    
   if(useHCAL_) 
    {
     evtplaneOutput->push_back(hcalPlane);
     evtplaneOutput->push_back(hcalpPlane);
     evtplaneOutput->push_back(hcalmPlane);
     evtplaneOutput->push_back(ethcalPlane);
     evtplaneOutput->push_back(ethcalpPlane);
     evtplaneOutput->push_back(ethcalmPlane);
    }
    
   if(useECAL_ && useHCAL_) 
    { 
     evtplaneOutput->push_back(caloPlane);
     evtplaneOutput->push_back(hfPlane);
     evtplaneOutput->push_back(hfpPlane);
     evtplaneOutput->push_back(hfmPlane);
     evtplaneOutput->push_back(calohfPlane);
     evtplaneOutput->push_back(calopPlane);
     evtplaneOutput->push_back(calomPlane);
     evtplaneOutput->push_back(calohfpPlane);
     evtplaneOutput->push_back(calohfmPlane);
     evtplaneOutput->push_back(etcaloPlane);
     evtplaneOutput->push_back(ethfPlane);
     evtplaneOutput->push_back(ethfpPlane);
     evtplaneOutput->push_back(ethfmPlane);
     evtplaneOutput->push_back(etcalohfPlane);
     evtplaneOutput->push_back(etcalopPlane);
     evtplaneOutput->push_back(etcalomPlane);
     evtplaneOutput->push_back(etcalohfpPlane);
     evtplaneOutput->push_back(etcalohfmPlane);    
     evtplaneOutput->push_back(hf1Plane);
     evtplaneOutput->push_back(hfp1Plane);
     evtplaneOutput->push_back(hfm1Plane);
     evtplaneOutput->push_back(ethf1Plane);
     evtplaneOutput->push_back(ethfp1Plane);
     evtplaneOutput->push_back(ethfm1Plane);
    
     }

   iEvent.put(evtplaneOutput, "recoLevel");


// cout << "  "<< planeA << endl;
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
EvtPlaneProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EvtPlaneProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EvtPlaneProducer);
