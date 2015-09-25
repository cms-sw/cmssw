
#include "GeneratorInterface/GenFilters/interface/MCProcessRangeFilter.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


MCProcessRangeFilter::MCProcessRangeFilter(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared"))),
minProcessID(iConfig.getUntrackedParameter("MinProcessID",0)),
maxProcessID(iConfig.getUntrackedParameter("MaxProcessID",500)),
pthatMin(iConfig.getUntrackedParameter("MinPthat",0)),
pthatMax(iConfig.getUntrackedParameter("MaxPthat",14000))
{

}


MCProcessRangeFilter::~MCProcessRangeFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to skim the data  ------------
bool MCProcessRangeFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);

   const HepMC::GenEvent * myGenEvent = evt->GetEvent();

    
    // do the selection -- processID 0 is always accepted
  
    if (myGenEvent->signal_process_id() > minProcessID && myGenEvent->signal_process_id() < maxProcessID) {
    
      if ( myGenEvent->event_scale() > pthatMin &&  myGenEvent->event_scale() < pthatMax ) { 
          accepted = true; 
      }  

    } 
    
    
   if (accepted){ return true; } else {return false;}

}

