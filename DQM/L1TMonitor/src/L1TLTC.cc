/*
 * \file L1TLTC.cc
 *
 * $Date: 2008/03/20 19:38:25 $
 * $Revision: 1.9 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TLTC.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
using namespace edm;

L1TLTC::L1TLTC(const ParameterSet& ps)
 {

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TLTC: constructor...." << endl;


  dbe = NULL;
  if ( ps.getUntrackedParameter<bool>("DQMStore", false) ) 
  {
    dbe = Service<DQMStore>().operator->();
    dbe->setVerbose(0);
  }

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if ( dbe !=NULL ) {
    dbe->setCurrentFolder("L1T/L1TLTC");
  }


}

L1TLTC::~L1TLTC()
{
}

void L1TLTC::beginJob(void)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1TLTC");
    dbe->rmdir("L1T/L1TLTC");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1T/L1TLTC");
    h1 = dbe->book1D("Bunch", "Bunch Number", 100, -0.5, 5000.) ;
    h2 = dbe->book1D("Orbit", "Orbit Number", 100, -0.5, 100000. ) ;
    h3 = dbe->book1D("Triggers", "Triggers", 8, -0.5, 7.5 ) ;

    overlaps = dbe->book2D("olaps", "Trigger Overlaps", 8, -0.5, 7.5 ,
			 8, -0.5, 7.5);
			 
    n_inhibit    = dbe->bookInt("n_inhibit");
    run          = dbe->bookInt("run");
    event        = dbe->bookInt("event");
    gps_time     = dbe->bookInt("gps_time");
  }  
}


void L1TLTC::endJob(void)
{
  if(verbose_) cout << "L1TLTC: end job...." << endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TLTC::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TLTC: analyze...." << endl;
  Handle< LTCDigiCollection > digis ;
  e.getByType(digis);
  
  if (!digis.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find LTCDigiCollection ";
    return;
  }

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

}

