
#include "GeneratorInterface/GenFilters/interface/HINEcalGenEvtSelector.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;

bool selectParticle(HepMC::GenParticle* par, int status, int pdg /*Absolute*/, double ptMin, double etaMax){
  return (par->status() == status && abs(par->pdg_id()) == pdg && par->momentum().perp() > ptMin && fabs(par->momentum().eta()) < etaMax);
}

HINEcalGenEvtSelector::HINEcalGenEvtSelector(const edm::ParameterSet& iConfig) :
  token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",std::string("generator"))))
{
  //now do what ever initialization is needed
  partonId_ = iConfig.getParameter<vector<int> >("partons");
  partonStatus_ = iConfig.getParameter<vector<int> >("partonStatus");
  partonPt_ = iConfig.getParameter<vector<double> >("partonPt");

  particleId_ = iConfig.getParameter<vector<int> >("particles");
  particleStatus_ = iConfig.getParameter<vector<int> >("particleStatus");
  particlePt_ = iConfig.getParameter<vector<double> >("particlePt");

  etaMax_ = iConfig.getParameter<double>("etaMax");

  int id = partonId_.size();
  int st = partonStatus_.size();
  int pt = partonPt_.size();

  if(partonId_.size() != partonStatus_.size() || partonId_.size() != partonPt_.size()){
    throw edm::Exception(edm::errors::LogicError)<<id<<st<<pt<<endl;
  }

  id = particleId_.size();
  st = particleStatus_.size();
  pt = particlePt_.size();

  if(particleId_.size() != particleStatus_.size() || particleId_.size() != particlePt_.size()){
    throw edm::Exception(edm::errors::LogicError)<<id<<st<<pt<<endl;
  }
}


HINEcalGenEvtSelector::~HINEcalGenEvtSelector()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool HINEcalGenEvtSelector::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);

   const HepMC::GenEvent * myGenEvent = evt->GetEvent();
   HepMC::GenEvent::particle_const_iterator begin = myGenEvent->particles_begin();
   HepMC::GenEvent::particle_const_iterator end = myGenEvent->particles_end();

   bool foundParticle = false;
   bool foundParton = false;

   HepMC::GenEvent::particle_const_iterator it = begin;
   while((!foundParton || !foundParticle) && it != end){
     for(unsigned i = 0; i < partonId_.size(); ++i){
       if(selectParticle(*it, partonStatus_[i], partonId_[i], partonPt_[i], etaMax_)) foundParton = true;
     }
     for(unsigned i = 0; i < particleId_.size(); ++i){
       if(selectParticle(*it, particleStatus_[i], particleId_[i], particlePt_[i], etaMax_)) foundParticle = true;
     }
     ++it;
   }

   return (foundParton && foundParticle);
}
