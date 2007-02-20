/*
 * \file L1TDTTPG.cc
 *
 * $Date: 2007/02/19 19:24:09 $
 * $Revision: 1.1 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TDTTPG.h"

using namespace std;
using namespace edm;

L1TDTTPG::L1TDTTPG(const ParameterSet& ps)
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
       "DT TPG phi bx", 10, -0.5, 9.5 ) ;  
    dttpgphwheel = dbe->book1D("DT TPG phi wheel number", 
       "DT TPG phi wheel number", 10, -0.5, 9.5 ) ;  
    dttpgphsector = dbe->book1D("DT TPG phi sector number", 
       "DT TPG phi sector number", 10, -0.5, 9.5 ) ;  
    dttpgphstation = dbe->book1D("DT TPG phi station number", 
       "DT TPG phi station number", 10, -0.5, 9.5 ) ;  
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

}

