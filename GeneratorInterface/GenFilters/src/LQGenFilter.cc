
#include "GeneratorInterface/GenFilters/interface/LQGenFilter.h"

LQGenFilter::LQGenFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  src_ = iConfig.getUntrackedParameter<edm::InputTag>("src",edm::InputTag("source"));
  
  eejj_     = iConfig.getParameter<bool>("eejj");
  enuejj_   = iConfig.getParameter<bool>("enuejj");
  nuenuejj_ = iConfig.getParameter<bool>("nuenuejj");
  
  mumujj_     = iConfig.getParameter<bool>("mumujj");
  munumujj_   = iConfig.getParameter<bool>("munumujj");
  numunumujj_ = iConfig.getParameter<bool>("numunumujj");
}


LQGenFilter::~LQGenFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
LQGenFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
                         
    if( abs((*p)->pdg_id()) != 42 ) continue; // skip if not a leptoquark

    if ( (*p)->end_vertex() ) {     
      for ( HepMC::GenVertex::particle_iterator des=(*p)->end_vertex()->particles_begin(HepMC::children);
            des != (*p)->end_vertex()->particles_end(HepMC::children); ++des ) {

          if ( abs((*des)->pdg_id()) == 11 ) ++ne;
          else if ( abs((*des)->pdg_id()) == 12 ) ++nnue;
          else if ( abs((*des)->pdg_id()) == 13 ) ++nmu;
          else if ( abs((*des)->pdg_id()) == 14 ) ++nnumu;
      }                            
    }
  }

  if (ne==2 && eejj_) return true;
  else if (ne==1 && nnue==1 && enuejj_) return true;
  else if (nnue==2 && nuenuejj_) return true;
  else if (nmu==2 && mumujj_) return true;
  else if (nmu==1 && nnumu==1 && munumujj_) return true;
  else if (nnumu==2 && numunumujj_) return true;
  else return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
LQGenFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LQGenFilter::endJob() {
}

