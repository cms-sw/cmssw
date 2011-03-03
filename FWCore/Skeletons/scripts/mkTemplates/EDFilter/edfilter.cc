// -*- C++ -*-
//
// Package:    fltrname
// Class:      fltrname
// 
/**\class fltrname fltrname.cc skelsubsys/fltrname/src/fltrname.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  John Doe
//         Created:  day-mon-xx
// RCS(Id)
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class fltrname : public edm::EDFilter {
   public:
      explicit fltrname(const edm::ParameterSet&);
      ~fltrname();

      static void fillDescriptions(ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual bool beginRun(Run&, EventSetup const&);
      virtual bool endRun(Run&, EventSetup const&);
      virtual bool beginLuminosityBlock(LuminosityBlock&, EventSetup const&);
      virtual bool endLuminosityBlock(LuminosityBlock&, EventSetup const&);

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
fltrname::fltrname(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


fltrname::~fltrname()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
fltrname::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
   return true;
}

// ------------ method called once each job just before starting event loop  ------------
void 
fltrname::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
fltrname::endJob() {
}

// ------------ method called when starting to processes a run  ------------
bool 
fltrname::beginRun(Run&, EventSetup const&)
{ 
  return true;
}

// ------------ method called when ending the processing of a run  ------------
bool 
fltrname::endRun(Run&, EventSetup const&)
{
  return true;
}

// ------------ method called when starting to processes a luminosity block  ------------
bool 
fltrname::beginLuminosityBlock(LuminosityBlock&, EventSetup const&)
{
  return true;
}

// ------------ method called when ending the processing of a luminosity block  ------------
bool 
fltrname::endLuminosityBlock(LuminosityBlock&, EventSetup const&)
{
  return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
fltrname::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(fltrname);
