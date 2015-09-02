#include "GeneratorInterface/GenFilters/interface/SingleLQGenFilter.h"

SingleLQGenFilter::SingleLQGenFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  src_ = iConfig.getUntrackedParameter<edm::InputTag>("src",edm::InputTag("source"));
  
  eej_     = iConfig.getParameter<bool>("eej");
  enuej_   = iConfig.getParameter<bool>("enuej");
  nuenuej_ = iConfig.getParameter<bool>("nuenuej");
  
  mumuj_     = iConfig.getParameter<bool>("mumuj");
  munumuj_   = iConfig.getParameter<bool>("munumuj");
  numunumuj_ = iConfig.getParameter<bool>("numunumuj");
}


SingleLQGenFilter::~SingleLQGenFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
SingleLQGenFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  int ne = 0;
  int nnue = 0;
  int nmu = 0;
  int nnumu = 0;

  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByLabel(src_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
        p != myGenEvent->particles_end(); ++p ) {

    //The pair prod. LQ Filter checks if the parent is an LQ (pdg_id()==42)
    //Single prod. uses LHE files from CalcHEP, no parent information
    //Instead, pythia8 status codes are used, only the 3 main outgoing particles have status==23

    if ((*p)->status()==23) {
      if ( abs((*p)->pdg_id()) == 11 ) ++ne;
      else if ( abs((*p)->pdg_id()) == 12 ) ++nnue;
      else if ( abs((*p)->pdg_id()) == 13 ) ++nmu;
      else if ( abs((*p)->pdg_id()) == 14 ) ++nnumu;
      
    }
  }
  
  //The same 4 filter-scenarios are provided, named with only one 'j', since single LQ decays produce only one jet
  //As in LQGenFilter.cc, no actual jet requirement is applied
  if (ne==2 && eej_) return true;
  else if (ne==1 && nnue==1 && enuej_) return true;
  else if (nnue==2 && nuenuej_) return true;
  else if (nmu==2 && mumuj_) return true;
  else if (nmu==1 && nnumu==1 && munumuj_) return true;
  else if (nnumu==2 && numunumuj_) return true;
  else return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
SingleLQGenFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SingleLQGenFilter::endJob() {
}

