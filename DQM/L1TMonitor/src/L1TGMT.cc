/*
 * \file L1TGMT.cc
 *
 * $Date: 2007/02/02 06:01:40 $
 * $Revision: 1.00 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TGMT.h"

using namespace std;
using namespace edm;

L1TGMT::L1TGMT(const ParameterSet& ps)
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TGMT: constructor...." << endl;

  logFile_.open("L1TGMT.log");

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
    dbe->setCurrentFolder("L1TMonitor/L1TGMT");
  }


}

L1TGMT::~L1TGMT()
{
}

void L1TGMT::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1TMonitor/L1TGMT");
    dbe->rmdir("L1TMonitor/L1TGMT");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1TMonitor/L1TGMT");
    
    gmtetavalue = dbe->book1D("CSC TF eta value", 
       "CSC TF eta value", 100, -2.5, 2.5 ) ;
    gmtphivalue = dbe->book1D("CSC TF phi value", 
       "CSC TF phi value", 100, 0.0, 6.2832 ) ;
    gmtptvalue = dbe->book1D("CSC TF pt value", 
       "CSC TF pt value", 160, -0.5, 159.5 ) ;
    gmtquality = dbe->book1D("CSC TF quality", 
       "CSC TF quality", 20, -0.5, 19.5 ) ;
    gmtcharge = dbe->book1D("CSC TF charge", 
       "CSC TF charge", 2, -1.5, 1.5 ) ;
    gmtntrack = dbe->book1D("CSC TF ntrack", 
       "CSC TF ntrack", 20, -0.5, 19.5 ) ;


  }  
}


void L1TGMT::endJob(void)
{
  if(verbose_) cout << "L1TGMT: end job...." << endl;
  LogInfo("L1TGMT") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TGMT::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TGMT: analyze...." << endl;


  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  e.getByType(pCollection);
  const L1MuGMTReadoutCollection* myCollection = pCollection.product();
  vector<L1MuGMTReadoutRecord> GMTRecordsVector;
  GMTRecordsVector = myCollection->getRecords();  
  if (verbose_)
  {
   std::cout << "GMT collection size: " << GMTRecordsVector.size()
   	    << std::endl;
  }

  int ngmttrack = 0;
  for( vector<L1MuGMTReadoutRecord>::iterator 
       RRItr = GMTRecordsVector.begin() ;
       RRItr != GMTRecordsVector.end() ;
       ++RRItr ) 
  {

    if (verbose_)
    {
    std::cout << "Readout Record " << RRItr->getBxInEvent()
   	    << std::endl;
    }

    vector<L1MuGMTExtendedCand> GMTCands = RRItr->getGMTCands();

    if (verbose_) 
    {
     std::cout << "GMTCands " << GMTCands.size()
   	    << std::endl;
    }

    for( vector<L1MuGMTExtendedCand>::iterator 
         ECItr = GMTCands.begin() ;
         ECItr != GMTCands.end() ;
         ++ECItr ) 
    {

      ngmttrack++;

      if (verbose_)
	{  
     std::cout << "GMTCand name " << ECItr->name()
   	    << std::endl;
	}

      gmtetavalue->Fill(ECItr->etaValue());
      if (verbose_)
	{     
     std::cout << "\tGMTCand eta value " << ECItr->etaValue()
   	    << std::endl;
	}

      gmtphivalue->Fill(ECItr->phiValue());
      if (verbose_)
	{     
     std::cout << "\tGMTCand phi value " << ECItr->phiValue()
   	    << std::endl;
	}

      gmtptvalue->Fill(ECItr->ptValue());
      if (verbose_)
	{     
     std::cout << "\tGMTCand pt value " << ECItr->ptValue()
   	    << std::endl;
	}

      gmtquality->Fill(ECItr->quality());
      if (verbose_)
	{     
     std::cout << "\tGMTCand quality " << ECItr->quality()
   	    << std::endl;
	}

      gmtcharge->Fill(ECItr->charge());
      if (verbose_)
	{     
     std::cout << "\tGMTCand charge " << ECItr->charge()
   	    << std::endl;
	}



    }
  }

      gmtntrack->Fill(ngmttrack);
      if (verbose_)
	{     
     std::cout << "\tGMTCand ntrack " << ngmttrack
   	    << std::endl;
	}
}

