#include <iostream>
#include "GeneratorInterface/HiGenCommon/interface/EcalGenEvtSelectorFrag.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "HepMC/GenVertex.h"
using namespace std;

EcalGenEvtSelectorFrag::EcalGenEvtSelectorFrag(const edm::ParameterSet& pset) : BaseHiGenEvtSelector(pset){

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

bool EcalGenEvtSelectorFrag::filter(HepMC::GenEvent *evt){
  
   HepMC::GenEvent::particle_const_iterator begin = evt->particles_begin();
   HepMC::GenEvent::particle_const_iterator end = evt->particles_end();

   bool foundParticle = false;
   bool foundParton = false;
 
   HepMC::GenEvent::particle_const_iterator it = begin;
   while((!foundParton || !foundParticle) && it != end){
     bool isFrag = false;
     bool isThisPhoton = false;

     foundParton = true; 
     /*for(unsigned i = 0; i < partonId_.size(); ++i){
       if(selectParticle(*it, partonStatus_[i], partonId_[i], partonPt_[i], etaMax_)) foundParton = true;
       }*/
     
     for(unsigned i = 0; i < particleId_.size(); ++i){
       if(selectParticle(*it, particleStatus_[i], particleId_[i], particlePt_[i], etaMax_)) isThisPhoton =true;
     }
     
     // Now you have to requre the partcile is "prompt", meaning its mom is parton
     
     if ( !((*it)->production_vertex()) ) {
       isFrag = false;
     }
     else {
       const HepMC::GenVertex* productionVertex = (*it)->production_vertex();
       
       size_t numberOfMothers = productionVertex->particles_in_size();
       if ( numberOfMothers <= 0 ) {
	 isFrag = false ;
	 //	 cout << "number of mothers = " << numberOfMothers << endl;
       }
       else {
	 //	 cout << "number of mothers = " << numberOfMothers << endl;
	 HepMC::GenVertex::particles_in_const_iterator motherIt = productionVertex->particles_in_const_begin();
	 for( ; motherIt != productionVertex->particles_in_const_end(); motherIt++) {
	   if ( fabs( (*motherIt)->pdg_id() ) <= 22 ) {
	     isFrag = true;
	   }
	 }
       }
     }
     
     if ( (isFrag == true) && ( isThisPhoton == true) ) {
       //cout << "Found fragmentation photon!!" << endl ;
       foundParticle = true;
     }
     
     ++it;
     
      
     
      
   }

   foundParton = true ; // We don't care of the parton
   return (foundParton && foundParticle);
}
