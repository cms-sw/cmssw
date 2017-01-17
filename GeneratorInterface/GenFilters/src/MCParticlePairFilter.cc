
//-ap #include "Configuration/CSA06Skimming/interface/MCParticlePairFilter.h"

#include "GeneratorInterface/GenFilters/interface/MCParticlePairFilter.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


MCParticlePairFilter::MCParticlePairFilter(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared"))),
particleCharge(iConfig.getUntrackedParameter("ParticleCharge",0)),
minInvMass(iConfig.getUntrackedParameter("MinInvMass", 0.)),
maxInvMass(iConfig.getUntrackedParameter("MaxInvMass", 14000.)),
minDeltaPhi(iConfig.getUntrackedParameter("MinDeltaPhi", 0.)),
maxDeltaPhi(iConfig.getUntrackedParameter("MaxDeltaPhi", 6.3)),
minDeltaR(iConfig.getUntrackedParameter("MinDeltaR",0.)),
maxDeltaR(iConfig.getUntrackedParameter("MaxDeltaR",10000.)),
betaBoost(iConfig.getUntrackedParameter("BetaBoost",0.))
{
   //here do whatever other initialization is needed
   vector<int> defpid1;
   defpid1.push_back(0);
   particleID1 = iConfig.getUntrackedParameter< vector<int> >("ParticleID1",defpid1);
   vector<int> defpid2;
   defpid2.push_back(0);
   particleID2 = iConfig.getUntrackedParameter< vector<int> >("ParticleID2",defpid2);
   vector<double> defptmin;   
   defptmin.push_back(0.);
   ptMin = iConfig.getUntrackedParameter< vector<double> >("MinPt", defptmin);
   vector<double> defpmin;
   defpmin.push_back(0.);
   pMin = iConfig.getUntrackedParameter< vector<double> >("MinP", defpmin);

   vector<double> defetamin;
   defetamin.push_back(-10.);
   etaMin = iConfig.getUntrackedParameter< vector<double> >("MinEta", defetamin);
   vector<double> defetamax ;
   defetamax.push_back(10.);
   etaMax = iConfig.getUntrackedParameter< vector<double> >("MaxEta", defetamax);
   vector<int> defstat ;
   defstat.push_back(0);
   status = iConfig.getUntrackedParameter< vector<int> >("Status", defstat);
   
   
    // check for correct size
    if (ptMin.size() != 2 
	|| pMin.size() != 2
	|| etaMin.size() != 2 
	|| etaMax.size() != 2 
	|| status.size() != 2 ) {
      cout << "WARNING: MCParticlePairFilter : size of some vectors not matching with 2!!" << endl;
    }
    
    // if ptMin size smaller than 2, fill up further with defaults
    if (2 > ptMin.size() ){
       vector<double> defptmin2 ;
       for (unsigned int i = 0; i < 2; i++){ defptmin2.push_back(0.);}
       ptMin = defptmin2;   
    } 
    // if pMin size smaller than 2, fill up further with defaults
    if (2 > pMin.size() ){
       vector<double> defpmin2 ;
       for (unsigned int i = 0; i < 2; i++){ defpmin2.push_back(0.);}
       pMin = defpmin2;
    }
    // if etaMin size smaller than 2, fill up further with defaults
    if (2 > etaMin.size() ){
       vector<double> defetamin2 ;
       for (unsigned int i = 0; i < 2; i++){ defetamin2.push_back(-10.);}
       etaMin = defetamin2;   
    } 
    // if etaMax size smaller than 2, fill up further with defaults
    if (2 > etaMax.size() ){
       vector<double> defetamax2 ;
       for (unsigned int i = 0; i < 2; i++){ defetamax2.push_back(10.);}
       etaMax = defetamax2;   
    }     
    // if status size smaller than 2, fill up further with defaults
    if (2 > status.size() ){
       vector<int> defstat2 ;
       for (unsigned int i = 0; i < 2; i++){ defstat2.push_back(0);}
       status = defstat2;   
    } 

}


MCParticlePairFilter::~MCParticlePairFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to skim the data  ------------
bool MCParticlePairFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);
   const double pi = 3.14159;

   vector<HepMC::GenParticle*> typeApassed;
   vector<HepMC::GenParticle*> typeBpassed;


   const HepMC::GenEvent * myGenEvent = evt->GetEvent();
     
   
   for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	 p != myGenEvent->particles_end(); ++p ) {
     
     // check for type A conditions
     bool gottypeAID = false;
     for(unsigned int j=0; j<particleID1.size(); ++j) {
       if(abs((*p)->pdg_id()) == abs(particleID1[j]) || particleID1[j] == 0) {
	 gottypeAID = true;
	 break;
       }
     }
     if(gottypeAID) {
       HepMC::FourVector mom = zboost((*p)->momentum());
       if ( mom.perp() > ptMin[0] && mom.rho() > pMin[0] && mom.eta() > etaMin[0] 
	    && mom.eta() < etaMax[0] && ((*p)->status() == status[0] || status[0] == 0)) { 
	 // passed A type conditions ...
	 // ... now check pair-conditions with B type passed particles
	 unsigned int i=0;
	 double deltaphi;
	 double phi1 = mom.phi();
	 double phi2;
	 double deltaeta;
	 double eta1 = mom.eta();
	 double eta2;
	 double deltaR;
	 //HepLorentzVector momentum1 = (*p)->momentum();
	 //HepLorentzVector totmomentum;
	 double tot_x= 0.;
	 double tot_y= 0.;
	 double tot_z= 0.;
	 double tot_e= 0.;
	 double invmass =0.;
	 int charge1 = 0;
	 int combcharge = 0;
	 while(!accepted && i<typeBpassed.size()) {
	   tot_x=mom.px();
	   tot_y=mom.py();
	   tot_z=mom.pz();
	   tot_e=mom.e();
	   charge1 = charge((*p)->pdg_id());
	   //totmomentum = momentum1 + typeBpassed[i]->momentum();
	   //invmass = totmomentum.m();
      HepMC::FourVector mom_i = zboost(typeBpassed[i]->momentum());
	   tot_x += mom_i.px();
	   tot_y += mom_i.py();
	   tot_z += mom_i.pz();
	   tot_e += mom_i.e();
	   invmass=sqrt(tot_e*tot_e-tot_x*tot_x-tot_y*tot_y-tot_z*tot_z);
	   combcharge = charge1 * charge(typeBpassed[i]->pdg_id());
	   if(invmass > minInvMass && invmass < maxInvMass) {
	     phi2 = mom_i.phi();
	     deltaphi = fabs(phi1-phi2);
	      if(deltaphi > pi) deltaphi = 2.*pi-deltaphi;
	      if(deltaphi > minDeltaPhi && deltaphi < maxDeltaPhi) {
		eta2 = mom_i.eta();
		deltaeta=fabs(eta1-eta2);
		deltaR = sqrt(deltaeta*deltaeta+deltaphi*deltaphi);
		if(deltaR > minDeltaR && deltaR < maxDeltaR) {
		  if(combcharge*particleCharge>=0) {
		    accepted = true;
		  }
		}
	      }
	   }
	   i++;
	 }	  
	 // if we found a matching pair quit the loop
	 if(accepted) break;
	 else{
	   typeApassed.push_back(*p);   // else remember the particle to have passed type A conditions
	 }
       }
     }
     
     // check for type B conditions
     
     bool gottypeBID = false;
     for(unsigned int j=0; j<particleID2.size(); ++j) {
       if(abs((*p)->pdg_id()) == abs(particleID2[j]) || particleID2[j] == 0) {
	 gottypeBID = true;
	 break;
       }
     }
     if(gottypeBID) {
       HepMC::FourVector mom = zboost((*p)->momentum());
       if ( mom.perp() > ptMin[1] && mom.rho() > pMin[1] && mom.eta() > etaMin[1] 
	    && mom.eta() < etaMax[1] && ((*p)->status() == status[1] || status[1] == 0)) { 
	 // passed B type conditions ...
	 // ... now check pair-conditions with A type passed particles vector
	 unsigned int i=0;
	 double deltaphi;
	 double phi1 = mom.phi();
	 double phi2;
	 double deltaeta;
	 double eta1 = mom.eta();
	 double eta2;
	 double deltaR;
	 //HepLorentzVector momentum1 = (*p)->momentum();
	 //HepLorentzVector totmomentum;
	 double tot_x= 0.;
	 double tot_y= 0.;
	 double tot_z= 0.;
	 double tot_e= 0.;
	 double invmass =0.;
	 int charge1 = 0;
	 int combcharge = 0;
	 while(!accepted && i<typeApassed.size()) {
	   if((*p) != typeApassed[i]) {
	     tot_x=mom.px();
	     tot_y=mom.py();
	     tot_z=mom.pz();
	     tot_e=mom.e();
	     charge1 = charge((*p)->pdg_id());
	     //totmomentum = momentum1 + typeApassed[i]->momentum();
        //invmass = totmomentum.m();
        HepMC::FourVector mom_i = zboost(mom_i);
	     tot_x += mom_i.px();
	     tot_y += mom_i.py();
	     tot_z += mom_i.pz();
	     tot_e += mom_i.e();
	     invmass=sqrt(tot_e*tot_e-tot_x*tot_x-tot_y*tot_y-tot_z*tot_z);
	     combcharge = charge1 * charge(typeApassed[i]->pdg_id());
	     if(invmass > minInvMass && invmass < maxInvMass) {
	       phi2 = mom_i.phi();
	       deltaphi = fabs(phi1-phi2);
	       if(deltaphi > pi) deltaphi = 2.*pi-deltaphi;
	       if(deltaphi > minDeltaPhi && deltaphi < maxDeltaPhi) {
		 eta2 = mom_i.eta();
		 deltaeta=fabs(eta1-eta2);
		 deltaR = sqrt(deltaeta*deltaeta+deltaphi*deltaphi);
		 if(deltaR > minDeltaR && deltaR < maxDeltaR) {
		   if(combcharge*particleCharge>=0) {
		     accepted = true;
		   }
		 }
	       }
	     }
	   }
	   i++;
	 }
	 // if we found a matching pair quit the loop
	 if(accepted) break;
	 else {
	   typeBpassed.push_back(*p);   // else remember the particle to have passed type B conditions
	 }
       }
     }
   }
   
   if (accepted){ return true; } else {return false;}
    
}

int MCParticlePairFilter::charge(const int& Id){

  
  //...Purpose: to give three times the charge for a particle/parton.

  //      ID     = particle ID
  //      hepchg = particle charge times 3

  int kqa,kq1,kq2,kq3,kqj,irt,kqx,kqn;
  int hepchg;


  int ichg[109]={-1,2,-1,2,-1,2,-1,2,0,0,-3,0,-3,0,-3,0,
-3,0,0,0,0,0,0,3,0,0,0,0,0,0,3,0,3,6,0,0,3,6,0,0,-1,2,-1,2,-1,2,0,0,0,0,
-3,0,-3,0,-3,0,0,0,0,0,-1,2,-1,2,-1,2,0,0,0,0,
-3,0,-3,0,-3,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


  //...Initial values. Simple case of direct readout.
  hepchg=0;
  kqa=abs(Id);
  kqn=kqa/1000000000%10;
  kqx=kqa/1000000%10;
  kq3=kqa/1000%10;
  kq2=kqa/100%10;
  kq1=kqa/10%10;
  kqj=kqa%10;
  irt=kqa%10000;

  //...illegal or ion
  //...set ion charge to zero - not enough information
  if(kqa==0 || kqa >= 10000000) {

    if(kqn==1) {hepchg=0;}
  }
  //... direct translation
  else if(kqa<=100) {hepchg = ichg[kqa-1];}
  //... KS and KL (and undefined)
  else if(kqj == 0) {hepchg = 0;}
  //C... direct translation
  else if(kqx>0 && irt<100)
    {
      hepchg = ichg[irt-1];
      if(kqa==1000017 || kqa==1000018) {hepchg = 0;}
      if(kqa==1000034 || kqa==1000052) {hepchg = 0;}
      if(kqa==1000053 || kqa==1000054) {hepchg = 0;}
      if(kqa==5100061 || kqa==5100062) {hepchg = 6;}
    }
  //...Construction from quark content for heavy meson, diquark, baryon.
  //...Mesons.
  else if(kq3==0)
    {
      hepchg = ichg[kq2-1]-ichg[kq1-1];
      //...Strange or beauty mesons.
      if((kq2==3) || (kq2==5)) {hepchg = ichg[kq1-1]-ichg[kq2-1];}
    }
  else if(kq1 == 0) {
    //...Diquarks.
    hepchg = ichg[kq3-1] + ichg[kq2-1];
  }

  else{
    //...Baryons
    hepchg = ichg[kq3-1]+ichg[kq2-1]+ichg[kq1-1];
  }

  //... fix sign of charge
  if(Id<0 && hepchg!=0) {hepchg = -1*hepchg;}

  // cout << hepchg<< endl;
  return hepchg;
}

HepMC::FourVector MCParticlePairFilter::zboost(const HepMC::FourVector& mom) {
   //Boost this Lorentz vector (from TLorentzVector::Boost)
   double b2 = betaBoost*betaBoost;
   double gamma = 1.0 / sqrt(1.0 - b2);
   double bp = betaBoost*mom.pz();
   double gamma2 = b2 > 0 ? (gamma - 1.0)/b2 : 0.0;

   return HepMC::FourVector(mom.px(), mom.py(), mom.pz() + gamma2*bp*betaBoost + gamma*betaBoost*mom.e(), gamma*(mom.e()+bp));
}
