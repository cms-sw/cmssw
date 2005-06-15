// -*- C++ -*-
//
// Package:    fltrname
// Class:      fltrname
// 
/**\class fltrname fltrname.cc skelsubsys/fltrname/src/fltrname.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
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
#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"
#include "FWCore/CoreFramework/interface/EDFilter.h"

#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class fltrname : public edm::EDFilter {
   public:
      explicit fltrname( const edm::ParameterSet& );
      ~fltrname();


      virtual bool filter( const Event&, const EventSetup& );
   private:
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
fltrname::fltrname( const edm::ParameterSet& iConfig )
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

// ------------ method called to produce the data  ------------
bool
fltrname::filter( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get( pSetup );
#endif
   return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(fltrname)
