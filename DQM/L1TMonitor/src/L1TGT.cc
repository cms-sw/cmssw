/*
 * \file L1TGT.cc
 *
 * $Date: 2007/07/25 15:05:21 $
 * $Revision: 1.9 $
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
    
    gttriggerdword = dbe->book1D("GT decision word", 
       "GT decision word", 128, -0.5, 3.5E+38 ) ;
    gttriggerdbits = dbe->book1D("GT decision bits", 
       "GT decision bits", 128, -0.5, 127.5 ) ;
    gttriggerdbitscorr = dbe->book2D("GT decision bit correlation","GT decision bit correlation", 128, -0.5, 127.5, 128, -0.5, 127.5 ) ;

    gtfdlbx = dbe->book1D("GT FDL Bx", "GT FDL Bx", 100, 0., 5000.) ;
    gtfdlevent = dbe->book1D("GT FDL Event", "GT FDL Event", 100, 0., 5000.) ;
    gtfdllocalbx = dbe->book1D("GT FDL local Bx", "GT FDL local Bx", 100, 0., 5000.) ;
    gtfdlbxinevent = dbe->book1D("GT FDL Bxinevent", "GT FDL Bxinevent", 100, 0., 5000.) ;
    gtfdlsize = dbe->book1D("GT FDL size", "GT FDL size",100,0., 100.);

    gtfeboardId = dbe->book1D("GT FE board Id", "GT FE board Id",100,0., 100.);
    gtferecordlength = dbe->book1D("GT FE record length", "GT FE record length",100,0., 100.);
    //    gtfebx = dbe->book1D("GT FE Bx","GT FE Bx",100, 0., 3500.);
    gtfebx = dbe->book1D("GT FE Bx","GT FE Bx",3500, 0., 3500.);
    gtfesetupversion = dbe->book1D("GT FE setup version","GT FE setup version",100,0.,100.);
    gtfeactiveboards = dbe->book1D("GT FE active boards","GT FE active boards",100,0.,1000.);
    gtfetotaltrigger = dbe->book1D("GT FE total triggers","GT FE total triggers",100,0.,5000.);
    gtfesize = dbe->book1D("GT FE size","GT FE size",100,0.,100.);
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

  //  int ntest = 5;
  //    gttest->Fill(ntest);
  //    if (verbose_)
  //	{     
  //   std::cout << "\tGT test " << ntest
  // 	    << std::endl;
  //	}

     Handle<L1GlobalTriggerReadoutRecord> myGTReadoutRecord;

     try {
      e.getByLabel(gtSource_,myGTReadoutRecord);
     }
     catch (...) {
     edm::LogInfo("L1TGT") << "can't find L1GlobalTriggerReadoutRecord with label "
			       << gtSource_.label() ;
     return;
     }

     L1GtfeWord mygtfeWord = myGTReadoutRecord->gtfeWord();
    if(verbose_) cout << "L1TGT: gtfe board Id " << mygtfeWord.boardId() << endl;
    gtfeboardId->Fill(mygtfeWord.boardId());
    if(verbose_) cout << "L1TGT: gtfe record length " << mygtfeWord.recordLength() << endl;
    gtferecordlength->Fill(mygtfeWord.recordLength());
    if(verbose_) cout << "L1TGT: gtfe bxNr " << mygtfeWord.bxNr() << endl;
    gtfebx->Fill(mygtfeWord.bxNr());
    if(verbose_) cout << "L1TGT: gtfe setupVersion " << mygtfeWord.setupVersion() << endl;
    gtfesetupversion->Fill(mygtfeWord.setupVersion());
    if(verbose_) cout << "L1TGT: gtfe active boards " << mygtfeWord.activeBoards() << endl;
    gtfeactiveboards->Fill(mygtfeWord.activeBoards());
   if(verbose_) cout << "L1TGT: gtfe totalTriggerNr " << mygtfeWord.totalTriggerNr() << endl;
   gtfetotaltrigger->Fill(mygtfeWord.totalTriggerNr());
   if(verbose_) cout << "L1TGT: gtfe size " << mygtfeWord.getSize() << endl;
   gtfesize->Fill(mygtfeWord.getSize());

   if (1)
     {
     L1GtPsbWord mygtPsbWord = myGTReadoutRecord->gtPsbWord(0,0);
     //   boost::uint16_t boardIdValue,
     //   int bxInEventValue,
     //   boost::uint16_t bxNrValue,
     //   boost::uint32_t eventNrValue,
     //   boost::uint16_t aDataValue[NumberAData],
     //   boost::uint16_t bDataValue[NumberBData],
     //   boost::uint16_t localBxNrValue
   if(verbose_) cout << "L1TGT: gtpsb board Id " << mygtPsbWord.boardId() << endl;
   if(verbose_) cout << "L1TGT: gtpsb bxInEvent " << mygtPsbWord.bxInEvent() << endl;
   if(verbose_) cout << "L1TGT: gtpsb bxNr " << mygtPsbWord.bxNr() << endl;
   if(verbose_) cout << "L1TGT: gtpsb eventNr " << mygtPsbWord.eventNr() << endl;
   if(verbose_) cout << "L1TGT: gtpsb localBxNr " << mygtPsbWord.localBxNr() << endl;
   if(verbose_) cout << "L1TGT: gtpsb size " << mygtPsbWord.getSize() << endl;
   for (int iA = 0; iA< mygtPsbWord.NumberAData; iA++)
     {
       if(verbose_) cout << "L1TGT: gtpsb AData " << iA << "\t" << mygtPsbWord.aData(iA) << endl;
     }
   for (int iB = 0; iB< mygtPsbWord.NumberBData; iB++)
     {
       if(verbose_) cout << "L1TGT: gtpsb BData " << iB << "\t" << mygtPsbWord.bData(iB) << endl;
}

     }

   if (1)
     {
     L1GtFdlWord mygtFdlWord = myGTReadoutRecord->gtFdlWord();
   if(verbose_) cout << "L1TGT: gtfdl board Id " << mygtFdlWord.boardId() << endl;
   if(verbose_) cout << "L1TGT: gtfdl bxInEvent " << mygtFdlWord.bxInEvent() << endl;
   gtfdlbxinevent->Fill(mygtFdlWord.bxInEvent());
   if(verbose_) cout << "L1TGT: gtfdl bxNr " << mygtFdlWord.bxNr() << endl;
   gtfdlbx->Fill(mygtFdlWord.bxNr());
  if(verbose_) cout << "L1TGT: gtfdl eventNr " << mygtFdlWord.eventNr() << endl;
  gtfdlevent->Fill(mygtFdlWord.eventNr());
   if(verbose_) cout << "L1TGT: gtfdl local BxNr " << mygtFdlWord.localBxNr() << endl;
   gtfdllocalbx->Fill(mygtFdlWord.localBxNr());
   if(verbose_) cout << "L1TGT: gtfdl size " << mygtFdlWord.getSize() << endl;
   gtfdlsize->Fill(mygtFdlWord.getSize());
     



     /// get Global Trigger decision and the decision word
     DecisionWord gtDecisionWord = myGTReadoutRecord->decisionWord();
     // decisionword is a vector of bools, loop through the vector and
     // accumulate triggers
     int dbitNumber = 0;
     int dword = 0;
     for( DecisionWord::const_iterator GTdbitItr =  gtDecisionWord.begin() ;
	  GTdbitItr != gtDecisionWord.end() ;
	  ++GTdbitItr ) 
     {
       if (*GTdbitItr)
       {
        gttriggerdbits->Fill(dbitNumber);
        dword += pow(2.0,dbitNumber);     
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
     gttriggerdword->Fill(dword);
     }
}
