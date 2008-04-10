#include "GeneratorInterface/AlpgenInterface/interface/AlpgenEmptyEventFilter.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

AlpgenEmptyEventFilter::AlpgenEmptyEventFilter(const edm::ParameterSet& ppp) 
{}

AlpgenEmptyEventFilter::~AlpgenEmptyEventFilter() 
{}

bool
AlpgenEmptyEventFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool statusOK = true;
   
   Handle<HepMCProduct> evt;
   iEvent.getByType(evt);
   // suggested by Carsten Hof to fix memory leak
   // HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
   const HepMC::GenEvent * myGenEvent = evt->GetEvent();

   if(myGenEvent->particles_empty()) // no particles in the event
     statusOK = false;
   return statusOK;
}


void 
AlpgenEmptyEventFilter::beginJob(const edm::EventSetup&)
{
}

void 
AlpgenEmptyEventFilter::endJob() {
}

//define this as a plug-in
