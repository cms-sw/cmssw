#include <iostream>
#include "GeneratorInterface/HiGenCommon/interface/EcalGenEvtSelector.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace std;

EcalGenEvtSelector::EcalGenEvtSelector(const edm::ParameterSet& pset) : BaseHiGenEvtSelector(pset){

   partonId_ = pset.getParameter<vector<int> >("partons");
   partonStatus_ = pset.getParameter<vector<int> >("partonStatus");
   partonPt_ = pset.getParameter<vector<double> >("partonPt");

   particleId_ = pset.getParameter<vector<int> >("particles");
   particleStatus_ = pset.getParameter<vector<int> >("particleStatus");
   particlePt_ = pset.getParameter<vector<double> >("particlePt");
   
   etaMax_ = pset.getParameter<double>("etaMax");
   
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

bool EcalGenEvtSelector::filter(HepMC::GenEvent *evt){
   HepMC::GenEvent::particle_const_iterator begin = evt->particles_begin();
   HepMC::GenEvent::particle_const_iterator end = evt->particles_end();

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
