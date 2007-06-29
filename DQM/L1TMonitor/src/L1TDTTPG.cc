/*
 * \file L1TDTTPG.cc
 *
 * $Date: 2007/02/21 22:10:31 $
 * $Revision: 1.3 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TDTTPG.h"

using namespace std;
using namespace edm;

L1TDTTPG::L1TDTTPG(const ParameterSet& ps)
  : dttpgSource_( ps.getParameter< InputTag >("dttpgSource") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TDTTPG: constructor...." << endl;

  logFile_.open("L1TDTTPG.log");

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
    dbe->setCurrentFolder("L1TMonitor/L1TDTTPG");
  }


}

L1TDTTPG::~L1TDTTPG()
{
}

void L1TDTTPG::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1TMonitor/L1TDTTPG");
    dbe->rmdir("L1TMonitor/L1TDTTPG");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1TMonitor/L1TDTTPG");

    dttpgphbx = dbe->book1D("DT TPG phi bx", 
       "DT TPG phi bx", 25, -0.5, 24.5 ) ;  
    dttpgphwheel = dbe->book1D("DT TPG phi wheel number", 
       "DT TPG phi wheel number", 5, -2.5, 2.5 ) ;  
    dttpgphsector = dbe->book1D("DT TPG phi sector number", 
       "DT TPG phi sector number", 11, -0.5, 10.5 ) ;  
    dttpgphstation = dbe->book1D("DT TPG phi station number", 
       "DT TPG phi station number", 4, 0.5, 4.5 ) ;  
    dttpgphphi = dbe->book1D("DT TPG phi", 
       "DT TPG phi", 20, -0.5, 19.5 ) ;  
    dttpgphphiB = dbe->book1D("DT TPG phiB", 
       "DT TPG phiB", 20, -0.5, 19.5 ) ;  
    dttpgphquality = dbe->book1D("DT TPG phi quality", 
       "DT TPG phi quality", 100, -0.5, 99.5 ) ;  
    dttpgphts2tag = dbe->book1D("DT TPG phi Ts2Tag", 
       "DT TPG phi Ts2Tag", 10, -0.5, 9.5 ) ;  
    dttpgphbxcnt = dbe->book1D("DT TPG phi BxCnt", 
       "DT TPG phi BxCnt", 10, -0.5, 9.5 ) ;  
    dttpgphntrack = dbe->book1D("DT TPG phi ntrack", 
       "DT TPG phi ntrack", 20, -0.5, 19.5 ) ;  

    dttpgthbx = dbe->book1D("DT TPG theta bx", 
       "DT TPG theta bx", 10, -0.5, 9.5 ) ;  
    dttpgthwheel = dbe->book1D("DT TPG theta wheel number", 
       "DT TPG theta wheel number", 10, -0.5, 9.5 ) ;  
    dttpgthsector = dbe->book1D("DT TPG theta sector number", 
       "DT TPG theta sector number", 10, -0.5, 9.5 ) ;  
    dttpgthstation = dbe->book1D("DT TPG theta station number", 
       "DT TPG theta station number", 10, -0.5, 9.5 ) ;  
    dttpgththeta = dbe->book1D("DT TPG theta", 
       "DT TPG theta", 20, -0.5, 19.5 ) ;  
    dttpgthquality = dbe->book1D("DT TPG theta quality", 
       "DT TPG theta quality", 100, -0.5, 99.5 ) ;  
    dttpgthntrack = dbe->book1D("DT TPG theta ntrack", 
       "DT TPG theta ntrack", 20, -0.5, 19.5 ) ;  
  }  
}


void L1TDTTPG::endJob(void)
{
  if(verbose_) cout << "L1TDTTPG: end job...." << endl;
  LogInfo("L1TDTTPG") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TDTTPG::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TDTTPG: analyze...." << endl;

  edm::Handle<L1MuDTChambPhContainer > myL1MuDTChambPhContainer;  
  e.getByLabel(dttpgSource_,myL1MuDTChambPhContainer);
  edm::Handle<L1MuDTChambThContainer > myL1MuDTChambThContainer;  
  e.getByLabel(dttpgSource_,myL1MuDTChambThContainer);
  int ndttpgphtrack = 0;
  int ndttpgthtrack = 0; 
  for (int iwheel = -2; iwheel <=2; iwheel++)
    {
      for (int istation = 1; istation <=4; istation++)
	{
	  for (int isector = 0; isector <=11; isector++)
	    {
	      for (int ibx = 0; ibx<=100; ibx++)
		{
             
		  L1MuDTChambPhDigi* const myDigi = myL1MuDTChambPhContainer->chPhiSegm1(iwheel, istation, isector, ibx);
                  if (myDigi)
		    {

		      ndttpgphtrack++;
		      dttpgphwheel->Fill(myDigi->whNum());
		      if (verbose_)
			{
		      cout << "DTTPG phi wheel number " << myDigi->whNum() << endl;
			}
                      dttpgphstation->Fill(myDigi->stNum());
		      if (verbose_)
			{   
 cout << "DTTPG phi station number " << myDigi->stNum() << endl;
			}
		      dttpgphsector->Fill(myDigi->scNum());
		      if (verbose_)
			{
    cout << "DTTPG phi sector number " << myDigi->scNum() << endl;
			}
                      dttpgphbx->Fill(myDigi->bxNum());
		      if (verbose_)
			{
    cout << "DTTPG phi bx number " << myDigi->bxNum() << endl;
			}
                    dttpgphphi->Fill(myDigi->phi());
		      if (verbose_)
			{
    cout << "DTTPG phi phi " << myDigi->phi() << endl;
			}
                    dttpgphphiB->Fill(myDigi->phiB());
		      if (verbose_)
			{
    cout << "DTTPG phi phiB " << myDigi->phiB() << endl;
			}
                    dttpgphquality->Fill(myDigi->code());
		      if (verbose_)
			{
    cout << "DTTPG phi quality " << myDigi->code() << endl;
			}
                    dttpgphts2tag->Fill(myDigi->Ts2Tag());
		      if (verbose_)
			{
    cout << "DTTPG phi ts2tag " << myDigi->Ts2Tag() << endl;
			}
                    dttpgphbxcnt->Fill(myDigi->BxCnt());
		      if (verbose_)
			{
    cout << "DTTPG phi bxcnt " << myDigi->BxCnt() << endl;
			}
		    }

		  L1MuDTChambThDigi* const mythDigi = myL1MuDTChambThContainer->chThetaSegm(iwheel, istation, isector, ibx);
                  if (mythDigi)
		    {
		      ndttpgthtrack++;
		      dttpgthwheel->Fill(mythDigi->whNum());
		      if (verbose_)
			{
    cout << "DTTPG theta wheel number " << mythDigi->whNum() << endl;
			}
                      dttpgthstation->Fill(mythDigi->stNum());
		      if (verbose_)
			{   
    cout << "DTTPG theta station number " << mythDigi->stNum() << endl;
			}
		      dttpgthsector->Fill(mythDigi->scNum());
		      if (verbose_)
			{
    cout << "DTTPG theta sector number " << mythDigi->scNum() << endl;
			}
                      dttpgthbx->Fill(mythDigi->bxNum());
		      if (verbose_)
			{
    cout << "DTTPG theta bx number " << mythDigi->bxNum() << endl;
			}
		      for (int j = 0; j < 7; j++)
			{
                    dttpgththeta->Fill(mythDigi->position(j));
		      if (verbose_)
			{
    cout << "DTTPG theta position " << mythDigi->position(j) << endl;
			}
                    dttpgthquality->Fill(mythDigi->code(j));
		      if (verbose_)
			{
    cout << "DTTPG theta quality " << mythDigi->code(j) << endl;
			}
			}
		    }
		    
		}

	    }
	}
    }
                    dttpgphntrack->Fill(ndttpgphtrack);
		      if (verbose_)
			{
    cout << "DTTPG phi ntrack " << ndttpgphtrack << endl;
			}
                    dttpgthntrack->Fill(ndttpgthtrack);
		      if (verbose_)
			{
    cout << "DTTPG theta ntrack " << ndttpgthtrack << endl;
			}

}


