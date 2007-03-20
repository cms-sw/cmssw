
#include "IOMC/GeneratorInterface/interface/PythiaHLTSoupFilter.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


PythiaHLTSoupFilter::PythiaHLTSoupFilter(const edm::ParameterSet& iConfig) :
label_(iConfig.getUntrackedParameter("moduleLabel",std::string("source"))),
minptelectron(iConfig.getUntrackedParameter("MinPtElectron", 0.)),
minptmuon(iConfig.getUntrackedParameter("MinPtMuon", 0.)),
maxetaelectron(iConfig.getUntrackedParameter("MaxEtaElectron", 10.)),
maxetamuon(iConfig.getUntrackedParameter("MaxEtaMuon", 10.)),
minpttau(iConfig.getUntrackedParameter("MinPtTau", 0.)),
maxetatau(iConfig.getUntrackedParameter("MaxEtaTau", 10.))

{
   //now do what ever initialization is needed

}


PythiaHLTSoupFilter::~PythiaHLTSoupFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool PythiaHLTSoupFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByLabel(label_, evt);

    HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
    
    if(myGenEvent->signal_process_id() == 2) {
    
    for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
	  p != myGenEvent->particles_end(); ++p ) {
	  
 
	if ( abs((*p)->pdg_id()) == 11 
	     && (*p)->momentum().perp() > minptelectron
	     && abs((*p)->momentum().eta()) < maxetaelectron
	     && (*p)->status() == 1 ) { accepted = true; }

           
	     
	if ( abs((*p)->pdg_id()) == 13 
	     && (*p)->momentum().perp() > minptmuon
	     && abs((*p)->momentum().eta()) < maxetamuon
	     && (*p)->status() == 1 ) { accepted = true; }
	     
	 
	 if ( abs((*p)->pdg_id()) == 15 
	     && (*p)->momentum().perp() > minpttau
	     && abs((*p)->momentum().eta()) < maxetatau
	     && (*p)->status() == 3 ) { accepted = true; }         
	     
	  
      }

    } else { accepted = true; }


    delete myGenEvent; 


   if (accepted){
   return true; } else {return false;}

}

