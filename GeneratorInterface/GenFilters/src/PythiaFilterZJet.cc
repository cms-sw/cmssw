#include "GeneratorInterface/GenFilters/interface/PythiaFilterZJet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <iostream>
#include<list>
#include<vector>
#include<cmath>

PythiaFilterZJet::PythiaFilterZJet(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")))),
etaMuMax(iConfig.getUntrackedParameter<double>("MaxMuonEta", 2.5)),
ptZMin(iConfig.getUntrackedParameter<double>("MinZPt")),
ptZMax(iConfig.getUntrackedParameter<double>("MaxZPt")),
maxnumberofeventsinrun(iConfig.getUntrackedParameter<int>("MaxEvents",10000)){ 
  
  theNumberOfSelected = 0;
}


PythiaFilterZJet::~PythiaFilterZJet(){}


// ------------ method called to produce the data  ------------
bool PythiaFilterZJet::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){

//  if(theNumberOfSelected>=maxnumberofeventsinrun)   {
//    throw cms::Exception("endJob")<<"we have reached the maximum number of events ";
//  }

  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent * myGenEvent = evt->GetEvent();


 if(myGenEvent->signal_process_id() == 15 || myGenEvent->signal_process_id() == 30) {


  std::vector<const HepMC::GenParticle *> mu;

  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   p != myGenEvent->particles_end(); ++p ) {
   
    if ( std::abs((*p)->pdg_id())==13 && (*p)->status()==1 )
      mu.push_back(*p);
    if(mu.size()>1) break;
  }

  //***
  
  if(mu.size()>1){
    math::XYZTLorentzVector tot_mom(mu[0]->momentum());
    math::XYZTLorentzVector mom2(mu[1]->momentum());
    tot_mom += mom2;
    //    double ptZ= (mu[0]->momentum() + mu[1]->momentum()).perp();
    double ptZ = tot_mom.pt();
    if (ptZ > ptZMin && ptZ < ptZMax  && 
	std::abs(mu[0]->momentum().eta()) < etaMuMax &&
	std::abs(mu[1]->momentum().eta()) < etaMuMax) 
      accepted=true;
  }

  } else {
  // end of if(gammajetevent)
  return true;
  // accept all non-gammajet events
  }

  if (accepted) {
    theNumberOfSelected++;
    return true; 
  }
  else return false;

}

