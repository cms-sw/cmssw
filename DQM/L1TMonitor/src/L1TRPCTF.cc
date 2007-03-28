/*
 * \file L1TRPCTF.cc
 *
 * $Date: 2007/02/02 06:01:40 $
 * $Revision: 1.00 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TRPCTF.h"

using namespace std;
using namespace edm;

L1TRPCTF::L1TRPCTF(const ParameterSet& ps)
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TRPCTF: constructor...." << endl;

  logFile_.open("L1TRPCTF.log");

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
    dbe->setCurrentFolder("L1TMonitor/L1TRPCTF");
  }


}

L1TRPCTF::~L1TRPCTF()
{
}

void L1TRPCTF::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1TMonitor/L1TRPCTF");
    dbe->rmdir("L1TMonitor/L1TRPCTF");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1TMonitor/L1TRPCTF");
    
    rpctfetavalue = dbe->book1D("RPC TF eta value", 
       "RPC TF eta value", 100, -2.5, 2.5 ) ;
    rpctfphivalue = dbe->book1D("RPC TF phi value", 
       "RPC TF phi value", 100, 0.0, 6.2832 ) ;
    rpctfptvalue = dbe->book1D("RPC TF pt value", 
       "RPC TF pt value", 160, -0.5, 159.5 ) ;
    rpctfptpacked = dbe->book1D("RPC TF pt_packed", 
       "RPC TF pt_packed", 160, -0.5, 159.5 ) ;
    rpctfquality = dbe->book1D("RPC TF quality", 
       "RPC TF quality", 20, -0.5, 19.5 ) ;
    rpctfchargevalue = dbe->book1D("RPC TF charge value", 
       "RPC TF charge value", 2, -1.5, 1.5 ) ;
    rpctfntrack = dbe->book1D("RPC TF ntrack", 
       "RPC TF ntrack", 20, -0.5, 19.5 ) ;
  }  
}


void L1TRPCTF::endJob(void)
{
  if(verbose_) cout << "L1TRPCTF: end job...." << endl;
  LogInfo("L1TRPCTF") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TRPCTF::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TRPCTF: analyze...." << endl;

  int nrpctftrack = 0;

  edm::Handle<std::vector<L1MuRegionalCand> > pRPCTFbtracks;  
  e.getByLabel("rpctrig","RPCb",pRPCTFbtracks);
  const std::vector<L1MuRegionalCand>* myRPCTFbTracks = 
    pRPCTFbtracks.product();
  std::auto_ptr<std::vector<L1MuRegionalCand> > L1RPCTFbTracks(new std::vector<L1MuRegionalCand>);
  L1RPCTFbTracks->insert(L1RPCTFbTracks->end(), myRPCTFbTracks->begin(), myRPCTFbTracks->end());
  

  std::cout << "RPCb TF collection size: " << L1RPCTFbTracks->size()
   	    << std::endl;
   for( vector<L1MuRegionalCand>::iterator 
        RPCTFItr =  L1RPCTFbTracks->begin() ;
        RPCTFItr != L1RPCTFbTracks->end() ;
        ++RPCTFItr ) 
   {
      nrpctftrack++;

     rpctfetavalue->Fill(RPCTFItr->etaValue());     
     if (verbose_)
       {
     std::cout << "RPC TF etavalue " << RPCTFItr->etaValue()  
   	    << std::endl;
       }

     rpctfphivalue->Fill(RPCTFItr->phiValue());     
     if (verbose_)
       {
     std::cout << "RPC TF phivalue " << RPCTFItr->phiValue()  
   	    << std::endl;
       }

     rpctfptvalue->Fill(RPCTFItr->ptValue());     
     if (verbose_)
       {
     std::cout << "RPC TF ptvalue " << RPCTFItr->ptValue()  
   	    << std::endl;
       }

     rpctfptpacked->Fill(RPCTFItr->pt_packed());     
     if (verbose_)
       {
     std::cout << "RPC TF pt_packed " << RPCTFItr->pt_packed()  
   	    << std::endl;
       }

     rpctfquality->Fill(RPCTFItr->quality());     
     if (verbose_)
       {
     std::cout << "RPC TF quality " << RPCTFItr->quality()  
   	    << std::endl;
       }

     rpctfchargevalue->Fill(RPCTFItr->chargeValue());     
     if (verbose_)
       {
     std::cout << "RPC TF charge value " << RPCTFItr->chargeValue()  
   	    << std::endl;
       }

    }

  edm::Handle<std::vector<L1MuRegionalCand> > pRPCTFftracks;  
  e.getByLabel("rpctrig","RPCf",pRPCTFftracks);
  const std::vector<L1MuRegionalCand>* myRPCTFfTracks = 
    pRPCTFftracks.product();
  std::auto_ptr<std::vector<L1MuRegionalCand> > L1RPCTFfTracks(new std::vector<L1MuRegionalCand>);
  L1RPCTFfTracks->insert(L1RPCTFfTracks->end(), myRPCTFfTracks->begin(), myRPCTFfTracks->end());
  

  std::cout << "RPCf TF collection size: " << L1RPCTFfTracks->size()
   	    << std::endl;
   for( vector<L1MuRegionalCand>::iterator 
        RPCTFItr =  L1RPCTFfTracks->begin() ;
        RPCTFItr != L1RPCTFfTracks->end() ;
        ++RPCTFItr ) 
   {
      nrpctftrack++;

     rpctfetavalue->Fill(RPCTFItr->etaValue());     
     if (verbose_)
       {
     std::cout << "RPC TF etavalue " << RPCTFItr->etaValue()  
   	    << std::endl;
       }

     rpctfphivalue->Fill(RPCTFItr->phiValue());     
     if (verbose_)
       {
     std::cout << "RPC TF phivalue " << RPCTFItr->phiValue()  
   	    << std::endl;
       }

     rpctfptvalue->Fill(RPCTFItr->ptValue());     
     if (verbose_)
       {
     std::cout << "RPC TF ptvalue " << RPCTFItr->ptValue()  
   	    << std::endl;
       }

     rpctfptpacked->Fill(RPCTFItr->pt_packed());     
     if (verbose_)
       {
     std::cout << "RPC TF pt_packed " << RPCTFItr->pt_packed()  
   	    << std::endl;
       }

     rpctfquality->Fill(RPCTFItr->quality());     
     if (verbose_)
       {
     std::cout << "RPC TF quality " << RPCTFItr->quality()  
   	    << std::endl;
       }

     rpctfchargevalue->Fill(RPCTFItr->chargeValue());     
     if (verbose_)
       {
     std::cout << "RPC TF charge value " << RPCTFItr->chargeValue()  
   	    << std::endl;
       }

    }

     rpctfntrack->Fill(nrpctftrack);     
     if (verbose_)
       {
     std::cout << "RPC TF ntrack " << nrpctftrack  
   	    << std::endl;
       }
}

