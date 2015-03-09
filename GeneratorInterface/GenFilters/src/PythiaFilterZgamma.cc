#include "GeneratorInterface/GenFilters/interface/PythiaFilterZgamma.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>
#include<list>
#include<vector>
#include<cmath>

PythiaFilterZgamma::PythiaFilterZgamma(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")))),
selProc(iConfig.getUntrackedParameter<int>("SelectProcess")),
ptElMin(iConfig.getUntrackedParameter<double>("MinElPt", 5.0)),
ptMuMin(iConfig.getUntrackedParameter<double>("MinMuPt", 3.0)),
ptPhotonMin(iConfig.getUntrackedParameter<double>("MinPhotPt", 5.0)),
etaElMax(iConfig.getUntrackedParameter<double>("MaxElecEta", 2.7)),
etaMuMax(iConfig.getUntrackedParameter<double>("MaxMuonEta", 2.4)),
etaPhotonMax(iConfig.getUntrackedParameter<double>("MaxPhotEta", 2.7))
{  
  theNumberOfSelected = 0;
}


PythiaFilterZgamma::~PythiaFilterZgamma(){}


// ------------ method called to produce the data  ------------
bool PythiaFilterZgamma::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){


  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent * myGenEvent = evt->GetEvent();

  std::vector<const HepMC::GenParticle *> el;
  std::vector<const HepMC::GenParticle *> mu;
  std::vector<const HepMC::GenParticle *> gam;

  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();  
               p != myGenEvent->particles_end(); ++p ) {
	          
    if(selProc == 1 ) {
      if ( std::abs((*p)->pdg_id())==11 && (*p)->status()==1 )
        el.push_back(*p);
      if(el.size()>1) break;
    } else if(selProc == 2 ) {
      if ( std::abs((*p)->pdg_id())==13 && (*p)->status()==1 )
        mu.push_back(*p);
      if(mu.size()>1) break;        
    }
      
  } // end of first particle loop for finding Z0 decays

  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();  
               p != myGenEvent->particles_end(); ++p ) {

    if ( std::abs((*p)->pdg_id())==22 && (*p)->status()==1 )
      gam.push_back(*p);
    if(gam.size()==1) break;

  } // end of second particle loop for finding gamma

  if(selProc == 1 ) {
    if(el.size()>1){
      double ptEl1 = el[0]->momentum().perp();
      double ptEl2 = el[1]->momentum().perp();
      double etaEl1 = el[0]->momentum().eta();
      double etaEl2 = el[1]->momentum().eta();
      if (ptEl1 > ptElMin && ptEl2 > ptElMin  && 
	std::abs(etaEl1) < etaElMax &&
	std::abs(etaEl2) < etaElMax) {
        if(gam.size()==1) {
	  double ptGam = gam[0]->momentum().perp();
          double etaGam = gam[0]->momentum().eta();
	  if(ptGam > ptPhotonMin && std::abs(etaGam) < etaPhotonMax)
                 accepted=true;
	} 	  
      } 
    } 
  } else if(selProc == 2 ) {

    if(mu.size()>1){
      double ptMu1 = mu[0]->momentum().perp();
      double ptMu2 = mu[1]->momentum().perp();
      double etaMu1 = mu[0]->momentum().eta();
      double etaMu2 = mu[1]->momentum().eta();
      if (ptMu1 > ptMuMin && ptMu2 > ptMuMin  && 
	std::abs(etaMu1) < etaMuMax &&
	std::abs(etaMu2) < etaMuMax) {
        if(gam.size()==1) {
	  double ptGam = gam[0]->momentum().perp();
          double etaGam = gam[0]->momentum().eta();
	  if(ptGam > ptPhotonMin && std::abs(etaGam) < etaPhotonMax)
                 accepted=true;
	} 	  
      } 
    } 
    
  }
/*
  if(accepted) {
    std::cout << "Accepted event Number: " << theNumberOfSelected 
              << "  of category " << selProc << std::endl;
    if(selProc == 1 ) { 
      std::cout << "Electon 1 pt, eta = " << el[0]->momentum().perp() << ", " 
                <<  el[0]->momentum().eta() << std::endl;
      std::cout << "Electon 2 pt, eta = " << el[1]->momentum().perp() << ", " 
                <<  el[1]->momentum().eta() << std::endl;
      std::cout << "Photon pt, eta = " << gam[0]->momentum().perp() << ", " 
                <<  gam[0]->momentum().eta() << std::endl;
    } else if(selProc == 2 ) {
      std::cout << "Muon 1 pt, eta = " << mu[0]->momentum().perp() << ", " 
                <<  mu[0]->momentum().eta() << std::endl;
      std::cout << "Muon 2 pt, eta = " << mu[1]->momentum().perp() << ", " 
                <<  mu[1]->momentum().eta() << std::endl;
      std::cout << "Photon pt, eta = " << gam[0]->momentum().perp() << ", " 
                <<  gam[0]->momentum().eta() << std::endl;

    }  
  }
*/
  if (accepted) {
    theNumberOfSelected++;
    return true; 
  }
  else return false;

}

