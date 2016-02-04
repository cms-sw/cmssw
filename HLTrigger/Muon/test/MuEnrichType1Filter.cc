// -*- C++ -*-
//
// Package:    MuEnrichType1Filter
// Class:      MuEnrichType1Filter
// 
/**\class MuEnrichType1Filter MuEnrichType1Filter.cc HLTrigger/MuEnrichType1Filter/src/MuEnrichType1Filter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Muriel VANDER DONCKT *:0
//         Created:  Fri Apr 27 17:05:15 CEST 2007
// $Id: MuEnrichType1Filter.cc,v 1.4 2009/10/15 11:50:20 fwyzard Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

 
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HLTrigger/Muon/test/MuEnrichType1Filter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuEnrichType1Filter::MuEnrichType1Filter(const edm::ParameterSet& iConfig)
{
  type = iConfig.getParameter<int>("type");
   //now do what ever initialization is needed
  nrejected=0;
  naccepted=0;
}


MuEnrichType1Filter::~MuEnrichType1Filter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
MuEnrichType1Filter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace HepMC;

   Handle< HepMCProduct > EvtHandle ;
   iEvent.getByLabel( "VtxSmeared", EvtHandle ) ;
   const GenEvent* Evt = EvtHandle->GetEvent() ;
   if (Evt != 0 ) {   
     edm::LogVerbatim ("MuEnrichFltr")  << "------------------------------";
     for ( HepMC::GenEvent::particle_const_iterator
	     part=Evt->particles_begin(); part!=Evt->particles_end(); ++part )
       {
	 if ( abs((*part)->pdg_id()) == 13 ) {
	   double pt=(*part)->momentum().perp();
           edm::LogVerbatim ("MuEnrichFltr")  << "Found a muon with pt"<<pt;
	   if ( pt>4 && type == 1) {
	     nrejected++;
	     return false;
	     break;
	   } else if ( pt>2 && pt<4 && type == 2 ) {
	     nrejected++;
	     return false;
	     break;
	   } else if ( pt>2 && pt<10 && type == 3 ) {
	     nrejected++;
	     return false;
	     break;
	   } 
	 }
       }
   }
   naccepted++;
   return true;
}

// ------------ method called once each job just before starting event loop  ------------
void 
MuEnrichType1Filter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuEnrichType1Filter::endJob() {
  edm::LogVerbatim ("MuEnrichFltr")  << "Total events"<<naccepted+nrejected;
  edm::LogVerbatim ("MuEnrichFltr")  << "Acccepted events"<<naccepted;
  edm::LogVerbatim ("MuEnrichFltr")  << "Rejected events"<<nrejected;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuEnrichType1Filter);
