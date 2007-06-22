/*
 * \file L1TRPCTF.cc
 *
 * $Date: 2007/02/22 19:43:53 $
 * $Revision: 1.3 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TRPCTF.h"

using namespace std;
using namespace edm;

L1TRPCTF::L1TRPCTF(const ParameterSet& ps)
  : rpctfbSource_( ps.getParameter< InputTag >("rpctfbSource") ),
  rpctffSource_( ps.getParameter< InputTag >("rpctffSource") )
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
    
    rpctfbetavalue = dbe->book1D("RPC TF barrel eta value", 
       "RPC TF barrel eta value", 100, -2.5, 2.5 ) ;
    rpctfbphivalue = dbe->book1D("RPC TF barrel phi value", 
       "RPC TF barrel phi value", 100, 0.0, 6.2832 ) ;
    rpctfbptvalue = dbe->book1D("RPC TF barrel pt value", 
       "RPC TF barrel pt value", 160, -0.5, 159.5 ) ;
    rpctfbptpacked = dbe->book1D("RPC TF barrel pt_packed", 
       "RPC TF barrel pt_packed", 160, -0.5, 159.5 ) ;
    rpctfbquality = dbe->book1D("RPC TF barrel quality", 
       "RPC TF barrel quality", 20, -0.5, 19.5 ) ;
    rpctfbchargevalue = dbe->book1D("RPC TF barrel charge value", 
       "RPC TF barrel charge value", 2, -1.5, 1.5 ) ;
    rpctfbntrack = dbe->book1D("RPC TF barrel ntrack", 
       "RPC TF barrel ntrack", 20, -0.5, 19.5 ) ;

    rpctffetavalue = dbe->book1D("RPC TF forward eta value", 
       "RPC TF forward eta value", 100, -2.5, 2.5 ) ;
    rpctffphivalue = dbe->book1D("RPC TF forward phi value", 
       "RPC TF forward phi value", 100, 0.0, 6.2832 ) ;
    rpctffptvalue = dbe->book1D("RPC TF forward pt value", 
       "RPC TF forward pt value", 160, -0.5, 159.5 ) ;
    rpctffptpacked = dbe->book1D("RPC TF forward pt_packed", 
       "RPC TF forward pt_packed", 160, -0.5, 159.5 ) ;
    rpctffquality = dbe->book1D("RPC TF forward quality", 
       "RPC TF forward quality", 20, -0.5, 19.5 ) ;
    rpctffchargevalue = dbe->book1D("RPC TF forward charge value", 
       "RPC TF forward charge value", 2, -1.5, 1.5 ) ;
    rpctffntrack = dbe->book1D("RPC TF forward ntrack", 
       "RPC TF forward ntrack", 20, -0.5, 19.5 ) ;
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

  int nrpctfbtrack = 0;

  edm::Handle<std::vector<L1MuRegionalCand> > pRPCTFbtracks;  
  try {
  e.getByLabel(rpctfbSource_,pRPCTFbtracks);
  }
  catch (...) {
    edm::LogInfo("L1RPCTF") << "can't find L1MuRegionalCand with label "
			       << rpctfbSource_.label() ;
    return;
  } 

   for( vector<L1MuRegionalCand>::const_iterator 
        RPCTFItr =  pRPCTFbtracks->begin() ;
        RPCTFItr != pRPCTFbtracks->end() ;
        ++RPCTFItr ) 
   {
      nrpctfbtrack++;

     rpctfbetavalue->Fill(RPCTFItr->etaValue());     
     if (verbose_)
       {
     std::cout << "RPC TF barrel etavalue " << RPCTFItr->etaValue()  
   	    << std::endl;
       }

     rpctfbphivalue->Fill(RPCTFItr->phiValue());     
     if (verbose_)
       {
     std::cout << "RPC TF barrel phivalue " << RPCTFItr->phiValue()  
   	    << std::endl;
       }

     rpctfbptvalue->Fill(RPCTFItr->ptValue());     
     if (verbose_)
       {
     std::cout << "RPC TF barrel ptvalue " << RPCTFItr->ptValue()  
   	    << std::endl;
       }

     rpctfbptpacked->Fill(RPCTFItr->pt_packed());     
     if (verbose_)
       {
     std::cout << "RPC TF barrel pt_packed " << RPCTFItr->pt_packed()  
   	    << std::endl;
       }

     rpctfbquality->Fill(RPCTFItr->quality());     
     if (verbose_)
       {
     std::cout << "RPC TF barrel quality " << RPCTFItr->quality()  
   	    << std::endl;
       }

     rpctfbchargevalue->Fill(RPCTFItr->chargeValue());     
     if (verbose_)
       {
     std::cout << "RPC TF barrel charge value " << RPCTFItr->chargeValue()  
   	    << std::endl;
       }

    }
     rpctfbntrack->Fill(nrpctfbtrack);     
     if (verbose_)
       {
     std::cout << "RPC TF barrel ntrack " << nrpctfbtrack  
   	    << std::endl;
       }

  edm::Handle<std::vector<L1MuRegionalCand> > pRPCTFftracks;  
 
 try {
  e.getByLabel(rpctffSource_,pRPCTFftracks);
  }
  catch (...) {
    edm::LogInfo("L1RPCTF") << "can't find L1MuRegionalCand with label "
			       << rpctffSource_.label() ;
    return;
  } 

  int nrpctfftrack = 0;
  for( vector<L1MuRegionalCand>::const_iterator 
        RPCTFItr =  pRPCTFftracks->begin() ;
        RPCTFItr != pRPCTFftracks->end() ;
        ++RPCTFItr ) 
   {
      nrpctfftrack++;

     rpctffetavalue->Fill(RPCTFItr->etaValue());     
     if (verbose_)
       {
     std::cout << "RPC TF forward etavalue " << RPCTFItr->etaValue()  
   	    << std::endl;
       }

     rpctffphivalue->Fill(RPCTFItr->phiValue());     
     if (verbose_)
       {
     std::cout << "RPC TF forward phivalue " << RPCTFItr->phiValue()  
   	    << std::endl;
       }

     rpctffptvalue->Fill(RPCTFItr->ptValue());     
     if (verbose_)
       {
     std::cout << "RPC TF forward ptvalue " << RPCTFItr->ptValue()  
   	    << std::endl;
       }

     rpctffptpacked->Fill(RPCTFItr->pt_packed());     
     if (verbose_)
       {
     std::cout << "RPC TF forward pt_packed " << RPCTFItr->pt_packed()  
   	    << std::endl;
       }

     rpctffquality->Fill(RPCTFItr->quality());     
     if (verbose_)
       {
     std::cout << "RPC TF forward quality " << RPCTFItr->quality()  
   	    << std::endl;
       }

     rpctffchargevalue->Fill(RPCTFItr->chargeValue());     
     if (verbose_)
       {
     std::cout << "RPC TF forward charge value " << RPCTFItr->chargeValue()  
   	    << std::endl;
       }

    }

     rpctffntrack->Fill(nrpctfftrack);     
     if (verbose_)
       {
     std::cout << "RPC TF forward ntrack " << nrpctfftrack  
   	    << std::endl;
       }
}

