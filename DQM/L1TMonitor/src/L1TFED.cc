/*
 * \file L1TFED.cc
 *
 * $Date: 2007/08/29 08:10:09 $
 * $Revision: 1.1 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TFED.h"

using namespace std;
using namespace edm;

L1TFED::L1TFED(const ParameterSet& ps)
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TFED: constructor...." << endl;

  logFile_.open("L1TFED.log");

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
    dbe->setCurrentFolder("L1TMonitor/L1TFED");
  }


}

L1TFED::~L1TFED()
{
}

void L1TFED::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1TMonitor/L1TFED");
    dbe->rmdir("L1TMonitor/L1TFED");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1TMonitor/L1TFED");
    
    fedtest = dbe->book1D("FED test", 
       "FED test", 128, -0.5, 127.5 ) ;	  
    hfedsize = dbe->book1D("fedsize","FED Size Distribution",100,0.,10000.);
    hfedprof = dbe->bookProfile("fedprof","FED Size by ID", 2048,0.,2048.,
				      0,0.,5000.);
    hindfed = new MonitorElement*[FEDNumbering::lastFEDId()];
    for(int i = 0; i<FEDNumbering::lastFEDId(); i++)
	  hindfed[i] = 0;
  }  
}


void L1TFED::endJob(void)
{
  if(verbose_) cout << "L1TFED: end job...." << endl;
  LogInfo("L1TFED") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TFED::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TFED: analyze...." << endl;

  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByType(rawdata);
  for (int i = 0; i<FEDNumbering::lastFEDId(); i++){
    const FEDRawData& data = rawdata->FEDData(i);
    if(size_t size=data.size()) {
       hfedsize->Fill(float(size));
       hfedprof->Fill(float(i),float(size));
       if(i<1024)
	 {
	  if(hindfed[i]==0)
	  {
	   DaqMonitorBEInterface *dbe = 
	   edm::Service<DaqMonitorBEInterface>().operator->();
	   dbe->setCurrentFolder("L1TMonitor/L1TFED/Details");
	   std::ostringstream os1;
	   std::ostringstream os2;
	   os1 << "fed" << i;
	   os2 << "FED #" << i << " Size Distribution";
	   hindfed[i] = dbe->book1D(os1.str(),os2.str(),100,0.,3.*size);
	   hindfed[i]->setResetMe(true);
	  }
	  hindfed[i]->Fill(float(size));
	 }
	}
       }
	  
}



