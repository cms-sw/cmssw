#include <iostream>
#include "GeneratorInterface/HiGenCommon/interface/HadronDecayGenEvtSelector.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace std;

HadronDecayGenEvtSelector::HadronDecayGenEvtSelector(const edm::ParameterSet& pset) : BaseHiGenEvtSelector(pset)
{

  hadronId_        = pset.getParameter<vector<int> >("hadrons");
  hadronStatus_    = pset.getParameter<vector<int> >("hadronStatus");
  hadronEtaMax_    = pset.getParameter<vector<double> >("hadronEtaMax");
  hadronEtaMin_    = pset.getParameter<vector<double> >("hadronEtaMin");
  hadronPMin_      = pset.getParameter<vector<double> >("hadronPMin");
  hadronPtMax_     = pset.getParameter<vector<double> >("hadronPtMax");
  hadronPtMin_     = pset.getParameter<vector<double> >("hadronPtMin");
  
  decayId_        = pset.getParameter<int>("decays");
  decayStatus_    = pset.getParameter<int>("decayStatus");
  decayEtaMax_    = pset.getParameter<double>("decayEtaMax");
  decayEtaMin_    = pset.getParameter<double>("decayEtaMin");
  decayPMin_      = pset.getParameter<double>("decayPMin");
  decayPtMax_     = pset.getParameter<double>("decayPtMax");
  decayPtMin_     = pset.getParameter<double>("decayPtMin");
  decayNtrig_     = pset.getParameter<int>("decayNtrig");

     
  int id     = hadronId_.size();
  int st     = hadronStatus_.size();
  int etamax = hadronEtaMax_.size();
  int etamin = hadronEtaMin_.size();
  int pmin   = hadronPMin_.size();
  int ptmax  = hadronPtMax_.size();
  int ptmin  = hadronPtMin_.size();
  
  if( id!=st || id!=etamax || id!=etamin || id!=ptmax || id!=ptmin || id!=pmin)
    {
      throw edm::Exception(edm::errors::LogicError)<<"Hadron selection parameters: "<<id<<st<<etamax<<etamin<<pmin<<ptmax<<ptmin<<endl;
    }
  

}


//____________________________________________________________________________________________
bool HadronDecayGenEvtSelector::filter(HepMC::GenEvent *evt)
{
  // loop over HepMC event, and search for  products of interest

  HepMC::GenEvent::particle_const_iterator begin = evt->particles_begin();
  HepMC::GenEvent::particle_const_iterator end   = evt->particles_end();
  
  bool foundHadron   = false;
  bool foundDecay    = false;
  
  int foundtrig = 0;
  HepMC::GenEvent::particle_const_iterator it = begin;
  while( (!foundHadron || !foundDecay) && it != end )
    {
      for(unsigned i = 0; i < hadronId_.size(); ++i)
	{
	  if( selectParticle(*it, 
			     hadronStatus_[i], hadronId_[i], 
			     hadronEtaMax_[i],hadronEtaMin_[i], 
			     hadronPMin_[i],
			     hadronPtMax_[i],hadronPtMin_[i]) ) foundHadron = true;
	}
     
      if( selectParticle(*it, 
			 decayStatus_, decayId_, 
			 decayEtaMax_,decayEtaMin_, 
			 decayPMin_,
			 decayPtMax_,decayPtMin_) ) foundtrig++;
      if(decayNtrig_ == foundtrig) foundDecay = true;
      
    ++it;
    }
  
  return (foundHadron && foundDecay);
}


//____________________________________________________________________________________________
