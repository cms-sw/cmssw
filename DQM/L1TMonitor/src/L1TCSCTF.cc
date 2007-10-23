/*
 * \file L1TCSCTF.cc
 *
 * $Date: 2007/02/22 19:43:53 $
 * $Revision: 1.3 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TCSCTF.h"

using namespace std;
using namespace edm;

L1TCSCTF::L1TCSCTF(const ParameterSet& ps)
  : csctfSource_( ps.getParameter< InputTag >("csctfSource") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TCSCTF: constructor...." << endl;

  logFile_.open("L1TCSCTF.log");

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
    dbe->setCurrentFolder("L1TMonitor/L1TCSCTF");
  }


}

L1TCSCTF::~L1TCSCTF()
{
}

void L1TCSCTF::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1TMonitor/L1TCSCTF");
    dbe->rmdir("L1TMonitor/L1TCSCTF");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1TMonitor/L1TCSCTF");
    
    csctfetavalue = dbe->book1D("CSC TF eta value", 
       "CSC TF eta value", 100, -2.5, 2.5 ) ;
    csctfphivalue = dbe->book1D("CSC TF phi value", 
       "CSC TF phi value", 100, 0.0, 6.2832 ) ;
    csctfptvalue = dbe->book1D("CSC TF pt value", 
       "CSC TF pt value", 160, -0.5, 159.5 ) ;
    csctfptpacked = dbe->book1D("CSC TF pt_packed", 
       "CSC TF pt_packed", 160, -0.5, 159.5 ) ;
    csctfquality = dbe->book1D("CSC TF quality", 
       "CSC TF quality", 20, -0.5, 19.5 ) ;
    csctfchargevalue = dbe->book1D("CSC TF charge value", 
       "CSC TF charge value", 2, -1.5, 1.5 ) ;
    csctfntrack = dbe->book1D("CSC TF ntrack", 
       "CSC TF ntrack", 20, -0.5, 19.5 ) ;


  }  
}


void L1TCSCTF::endJob(void)
{
  if(verbose_) cout << "L1TCSCTF: end job...." << endl;
  LogInfo("L1TCSCTF") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TCSCTF::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TCSCTF: analyze...." << endl;


  edm::Handle<std::vector<L1MuRegionalCand> > pCSCTFtracks;  
 

  try {
  e.getByLabel(csctfSource_,pCSCTFtracks);
  }
  catch (...) {
    edm::LogInfo("L1TCSCTF") << "can't find L1MuGMTRegionalCand with label "
			       << csctfSource_.label() ;
    return;
  }

  int ncsctftrack = 0;
   for( vector<L1MuRegionalCand>::const_iterator 
        CSCTFtrackItr =  pCSCTFtracks->begin() ;
        CSCTFtrackItr != pCSCTFtracks->end() ;
        ++CSCTFtrackItr ) 
   {

      ncsctftrack++;

     csctfetavalue->Fill(CSCTFtrackItr->etaValue());     
     if (verbose_)
       {
     std::cout << "CSC TF etavalue " << CSCTFtrackItr->etaValue()  
   	    << std::endl;
       }

     csctfphivalue->Fill(CSCTFtrackItr->phiValue());     
     if (verbose_)
       {
     std::cout << "CSC TF phivalue " << CSCTFtrackItr->phiValue()  
   	    << std::endl;
       }

     csctfptvalue->Fill(CSCTFtrackItr->ptValue());     
     if (verbose_)
       {
     std::cout << "CSC TF ptvalue " << CSCTFtrackItr->ptValue()  
   	    << std::endl;
       }

     csctfptpacked->Fill(CSCTFtrackItr->pt_packed());     
     if (verbose_)
       {
     std::cout << "CSC TF pt_packed " << CSCTFtrackItr->pt_packed()  
   	    << std::endl;
       }

     csctfquality->Fill(CSCTFtrackItr->quality());     
     if (verbose_)
       {
     std::cout << "CSC TF quality " << CSCTFtrackItr->quality()  
   	    << std::endl;
       }

     csctfchargevalue->Fill(CSCTFtrackItr->chargeValue());     
     if (verbose_)
       {
     std::cout << "CSC TF charge value " << CSCTFtrackItr->chargeValue()  
   	    << std::endl;
       }

    }

     csctfntrack->Fill(ncsctftrack);     
     if (verbose_)
       {
     std::cout << "CSC TF ntrack " << ncsctftrack  
   	    << std::endl;
       }

}

