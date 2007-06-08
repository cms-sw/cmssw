/*
 * \file L1TGT.cc
 *
 * $Date: 2007/02/22 19:43:53 $
 * $Revision: 1.2 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TGT.h"

using namespace std;
using namespace edm;

L1TGT::L1TGT(const ParameterSet& ps)
  : gtSource_( ps.getParameter< InputTag >("gtSource") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TGT: constructor...." << endl;

  logFile_.open("L1TGT.log");

  dbe = NULL;
  if ( ps.getUntrackedParameter<bool>("DaqMonitorBEInterface", false) ) 
  {
    dbe = Service<DaqMonitorBEInterface>().operator->();
    dbe->setVerbose(0);
  }

  monitorDaemon_ = false;
  if ( ps.getUntrackedParameter<bool>("MonitorDaemon", false) ) {
    Service<MonitorDaemon> daemon;
    daemon.operator->();
    monitorDaemon_ = true;
  }

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }
  else{
    outputFile_ = "L1TDQM.root";
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if ( dbe !=NULL ) {
    dbe->setCurrentFolder("L1TMonitor/L1TGT");
  }


}

L1TGT::~L1TGT()
{
}

void L1TGT::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1TMonitor/L1TGT");
    dbe->rmdir("L1TMonitor/L1TGT");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1TMonitor/L1TGT");
    
    gttest = dbe->book1D("GT test", 
       "GT test", 128, -0.5, 127.5 ) ;
    gttriggerdbits = dbe->book1D("GT decision bits", 
       "GT decision bits", 128, -0.5, 127.5 ) ;
    gttriggerdbitscorr = dbe->book2D("GT decision bit correlation","GT decision bit correlation", 128, -0.5, 127.5, 128, -0.5, 127.5 ) ;
  }  
}


void L1TGT::endJob(void)
{
  if(verbose_) cout << "L1TGT: end job...." << endl;
  LogInfo("L1TGT") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TGT::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TGT: analyze...." << endl;

  int ntest = 5;
      gttest->Fill(ntest);
      if (verbose_)
	{     
     std::cout << "\tGT test " << ntest
   	    << std::endl;
	}

     Handle<L1GlobalTriggerReadoutRecord> myGTReadoutRecord;
     e.getByLabel(gtSource_,myGTReadoutRecord);

     /// get Global Trigger decision and the decision word
     DecisionWord gtDecisionWord = myGTReadoutRecord->decisionWord();
     // decisionword is a vector of bools, loop through the vector and
     // accumulate triggers
     int dbitNumber = 0;
     for( DecisionWord::const_iterator GTdbitItr =  gtDecisionWord.begin() ;
	  GTdbitItr != gtDecisionWord.end() ;
	  ++GTdbitItr ) 
     {
       if (*GTdbitItr)
       {
        gttriggerdbits->Fill(dbitNumber);     

        int dbitNumber1 = 0;
        for( DecisionWord::const_iterator GTdbitItr1 =  
	       gtDecisionWord.begin() ; GTdbitItr1 != gtDecisionWord.end() ;
	     ++GTdbitItr1 ) 
        {
         if (*GTdbitItr1) gttriggerdbitscorr->Fill(dbitNumber,dbitNumber1);
         dbitNumber1++; 
        }
       if (verbose_)
	 { 
      cout << dbitNumber << "\t" << *GTdbitItr << endl;
	 } 
      dbitNumber++; 
       }
     }

}
