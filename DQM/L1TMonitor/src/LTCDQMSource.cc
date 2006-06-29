// -*- C++ -*-
//
// Package:    LTCDQMSource
// Class:      LTCDQMSource
// 
/**\class LTCDQMSource LTCDQMSource.cc DQMServices/LTCDQMSource/src/LTCDQMSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Sun
//         Created:  Wed May 24 11:58:16 EDT 2006
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

using namespace std ;

//
// class decleration
//

class LTCDQMSource : public edm::EDAnalyzer {
   public:
      explicit LTCDQMSource(const edm::ParameterSet&);
      ~LTCDQMSource();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(void);
   private:
      // ----------member data ---------------------------
      MonitorElement* h1;
      MonitorElement* h2;
      MonitorElement* h3;
//       MonitorElement* hlist[ 6 ] ;
      float XMIN; float XMAX;
      // event counter
      int counter;
      // back-end interface
      DaqMonitorBEInterface * dbe;
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
LTCDQMSource::LTCDQMSource(const edm::ParameterSet& iConfig)
{
   // get hold of back-end interface
   dbe = edm::Service<DaqMonitorBEInterface>().operator->();

   // now do what ever initialization is needed
   edm::Service<MonitorDaemon> daemon;
   daemon.operator->();

   // book some histograms here

   // create and cd into new folder
   dbe->setCurrentFolder("C1");
   h1 = dbe->book1D("histo", "Bunch Number", 100, 0., 5000.) ;
   h2 = dbe->book1D("histo2", "Orbit Number", 100, 0., 100000. ) ;
   h3 = dbe->book1D("histo3", "Triggers", 8, -0.5, 7.5 ) ;

//    hlist[ 0 ] = dbe->book1D("hlist0", "Trigger 0 Rate", 2, 0., 2. ) ;
//    hlist[ 1 ] = dbe->book1D("hlist1", "Trigger 1 Rate", 2, 0., 2. ) ;
//    hlist[ 2 ] = dbe->book1D("hlist2", "Trigger 2 Rate", 2, 0., 2. ) ;
//    hlist[ 3 ] = dbe->book1D("hlist3", "Trigger 3 Rate", 2, 0., 2. ) ;
//    hlist[ 4 ] = dbe->book1D("hlist4", "Trigger 4 Rate", 2, 0., 2. ) ;
//    hlist[ 5 ] = dbe->book1D("hlist5", "Trigger 5 Rate", 2, 0., 2. ) ;

//    // contents of h5 & h6 will be reset at end of monitoring cycle
//    h5->setResetMe(true);
//    h6->setResetMe(true);
}


LTCDQMSource::~LTCDQMSource()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//


void LTCDQMSource::endJob(void)
{
  dbe->save("ltcsource.root");  
}

// ------------ method called to produce the data  ------------
void
LTCDQMSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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

   Handle< LTCDigiCollection > digis ;
//  iEvent.getByLabel( "digis", digis ) ;
   iEvent.getByType( digis ) ;

   for( LTCDigiCollection::const_iterator digiItr = digis->begin() ;
	digiItr != digis->end() ;
	++digiItr )
   {
      h1->Fill( digiItr->bunchNumber() ) ;
      h2->Fill( digiItr->orbitNumber() ) ;

      for( int i = 0 ; i < 6 ; ++i )
      {
	 h3->Fill( i, ( digiItr->HasTriggered( i ) ? 1 : 0 ) ) ;
// 	 hlist[ i ]->Fill( digiItr->HasTriggered( i ) ? 1 : 0 ) ;
      }

      h3->Fill( 6, digiItr->ramTrigger() ) ;
      h3->Fill( 7, digiItr->vmeTrigger() ) ;
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(LTCDQMSource)
