// -*- C++ -*-
//
// Package:    L1TLTC
// Class:      L1TLTC
// 
/**\class L1TLTC L1TLTC.cc DQM/L1TMonitor/src/L1TLTC.cc

   Description: DQM Source for LTC data

   Implementation:
   <Notes on implementation>
*/
//
// Original Author:  Werner Sun
//         Created:  Wed May 24 11:58:16 EDT 2006
// $Id: LTCDQMSource.cc,v 1.6 2006/10/31 20:26:51 meschi Exp $
//
//


#include "DQM/L1TMonitor/interface/L1TLTC.h"

//
// constructors and destructor
//
L1TLTC::L1TLTC(const edm::ParameterSet& iConfig):
  nev_(0)
{
  // get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  // now do what ever initialization is needed
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  // book some histograms here

  // create and cd into new folder
  // Every filter unit will prepend this with a filter unit string
  // so to see all the LTC data you need to select
  // Collector/*/L1Trigger/LTC/*
  dbe->setCurrentFolder("L1Trigger/LTC");
  h1 = dbe->book1D("Bunch", "Bunch Number", 100, 0., 5000.) ;
  h2 = dbe->book1D("Orbit", "Orbit Number", 100, 0., 100000. ) ;
  h3 = dbe->book1D("Triggers", "Triggers", 8, -0.5, 7.5 ) ;

  overlaps = dbe->book2D("olaps", "Trigger Overlaps", 8, -0.5, 7.5 ,
			 8, -0.5, 7.5);
			 
  n_inhibit    = dbe->bookInt("n_inhibit");
  run          = dbe->bookInt("run");
  event        = dbe->bookInt("event");
  gps_time     = dbe->bookInt("gps_time");

  saveMe_ = iConfig.getUntrackedParameter<bool>("saveRootFile",false);
  rootFileName_ = iConfig.getUntrackedParameter<std::string>("RootFileName","ciccio.root");


  //    // contents of h5 & h6 will be reset at end of monitoring cycle
  //    h5->setResetMe(true);
  //    h6->setResetMe(true);
  std::cout << "L1TLTC: configured histograms." << std::endl;
}


L1TLTC::~L1TLTC()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
// void L1TLTC::beginRun()
// {
// }


void L1TLTC::endJob(void)
{
  std::cout << "L1TLTC: saw " << nev_ << " events. " << std::endl;
  if ( saveMe_ ) 
    dbe->save(rootFileName_);
  dbe->setCurrentFolder("L1Trigger/LTC");
  dbe->removeContents();
}

// ------------ method called to produce the data  ------------
void
L1TLTC::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  Handle< LTCDigiCollection > digis ;
  //  iEvent.getByLabel( "digis", digis ) ;
  iEvent.getByType( digis ) ;

  for( LTCDigiCollection::const_iterator digiItr = digis->begin() ;
       digiItr != digis->end() ;
       ++digiItr )
    {
      h1->Fill( digiItr->bunchNumber() ) ;
      h2->Fill( digiItr->orbitNumber() ) ;

      for( int i = 0 ; i < 6 ; ++i )	{
	h3->Fill( i, ( digiItr->HasTriggered( i ) ? 1 : 0 ) ) ;
      }

      h3->Fill( 6, digiItr->ramTrigger() ) ;
      h3->Fill( 7, digiItr->vmeTrigger() ) ;
      // overlaps
      unsigned int setbits = digiItr->externTriggerMask();
      // mock up the VME and RAM triggers
      if ( digiItr->ramTrigger() ) {
	setbits |= (0x1UL<<7);
      }
      if ( digiItr->vmeTrigger() ) {
	setbits |= (0x1UL<<8);
      }
      for ( int i = 0; i < 8; ++i ) {
	if ( setbits & (0x1UL<<i) ) {
	  for ( int j = i; j < 8; ++j ) {
	    if ( setbits & (0x1UL<<j) ) {
	      overlaps->Fill(i,j); // do both....
	      overlaps->Fill(j,i);
	    }
	  }
	}
      }
      // fill floats and ints
      n_inhibit->Fill(digiItr->triggerInhibitNumber());
      run      ->Fill(digiItr->runNumber());
      event    ->Fill(digiItr->eventNumber());
      gps_time ->Fill(digiItr->bstGpsTime());
    }
  ++nev_;
  //usleep(10000);
}

