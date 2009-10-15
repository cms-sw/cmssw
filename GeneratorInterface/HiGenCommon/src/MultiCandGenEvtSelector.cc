#include <iostream>
#include "GeneratorInterface/HiGenCommon/interface/MultiCandGenEvtSelector.h"

MultiCandGenEvtSelector::MultiCandGenEvtSelector(const edm::ParameterSet& iConfig)
   : BaseHiGenEvtSelector(iConfig)
{
   ptMin_ = iConfig.getParameter<double>("ptMin");
   etaMax_ = iConfig.getParameter<double>("etaMax");
   pdg_ = iConfig.getParameter<int>("pdg");
   st_ = iConfig.getParameter<int>("status");
   nTrig_ = iConfig.getParameter<int>("minimumCandidates");
}

bool MultiCandGenEvtSelector::filter(HepMC::GenEvent * evt){
   std::cout<<"Di Muon Fired"<<std::endl;
   
   int found = 0;
   HepMC::GenEvent::particle_const_iterator begin = evt->particles_begin();
   HepMC::GenEvent::particle_const_iterator end = evt->particles_end();
   for(HepMC::GenEvent::particle_const_iterator it = begin; it != end; ++it){
      if(selectParticle(*it, st_, pdg_, ptMin_, etaMax_)) found++;
      if(found == nTrig_) return true;      
   }
   
   return false;
}
