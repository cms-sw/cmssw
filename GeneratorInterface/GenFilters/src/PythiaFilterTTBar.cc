#include "GeneratorInterface/GenFilters/interface/PythiaFilterTTBar.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

PythiaFilterTTBar::PythiaFilterTTBar(const edm::ParameterSet& iConfig) : 
  token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared"))),
  decayType_(iConfig.getUntrackedParameter("decayType",1)),
  leptonFlavour_(iConfig.getUntrackedParameter("leptonFlavour",0))	     
{
}


PythiaFilterTTBar::~PythiaFilterTTBar()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
PythiaFilterTTBar::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  bool accept=false;

  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent * myGenEvent = evt->GetEvent();

  unsigned int iE=0, iMu=0, iTau=0;

  unsigned int iNuE=0, iNuMu=0, iNuTau=0;

  unsigned int iLep=0, iNu=0;


  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   p != myGenEvent->particles_end(); ++p ) {

    int pdgID = (*p)->pdg_id();
    
    int status = (*p)->status();

    if ( status == 3 ) {

      // count the final state leptons

      if (fabs(pdgID) == 11)
	iE++;

      if(fabs(pdgID) == 13)
	iMu++;
      
      if(fabs(pdgID) == 15)
	iTau++;

      // count the final state neutrinos
      
      if(fabs(pdgID) == 12)
	iNuE++;

      if(fabs(pdgID) == 14)
	iNuMu++;
      
      if(fabs(pdgID) == 16)
	iNuTau++;

    }

  }

  iLep = (iE+iMu+iTau);
  iNu = (iNuE+iNuMu+iNuTau);

  if (decayType_ == 1) { // semi-leptonic decay
    
    // l = e,mu,tau

    if (leptonFlavour_ == 0 && iLep == 1 && iNu == 1)
      accept=true;
    
    // l = e

    else if (leptonFlavour_ == 1 && iE == 1 && iNuE == 1 && iLep == 1 && iNu == 1)
      accept=true;

    // l = mu

    else if (leptonFlavour_ == 2 && iMu == 1 && iNuMu == 1 && iLep == 1 && iNu == 1)
      accept=true;
    
    // l = tau 

    else if (leptonFlavour_ == 3 && iTau == 1 && iNuTau == 1 && iLep == 1 && iNu == 1)
      accept=true;

  }

  else if (decayType_ == 2) { // di-leptonic decay (inclusive)

    if (iLep == 2 && iNu == 2) 
      accept=true;
    
  }
  
  else if (decayType_ == 3) { // fully-hadronic decay

    if (iLep == 0 && iNu == 0)
      accept=true;
  }

  else
    accept=false;
  
  
  return accept;
}

