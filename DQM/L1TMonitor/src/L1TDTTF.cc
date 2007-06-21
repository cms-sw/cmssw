/*
 * \file L1TDTTF.cc
 *
 * $Date: 2007/05/25 15:45:48 $
 * $Revision: 1.3 $
 * \author J. Berryhill
 * $Log$
 */

#include "DQM/L1TMonitor/interface/L1TDTTF.h"

using namespace std;
using namespace edm;

L1TDTTF::L1TDTTF(const ParameterSet& ps)
  : dttfSource_( ps.getParameter< InputTag >("dttfSource") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TDTTF: constructor...." << endl;

  logFile_.open("L1TDTTF.log");

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
    dbe->setCurrentFolder("L1TMonitor/L1TDTTF");
  }


}

L1TDTTF::~L1TDTTF()
{
}

void L1TDTTF::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = 0;
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1TMonitor/L1TDTTF");
    dbe->rmdir("L1TMonitor/L1TDTTF");
  }


  if ( dbe ) 
    {
      dbe->setCurrentFolder("L1TMonitor/L1TDTTF");
    
      dttfetavalue = dbe->book1D("DT_TF_eta value", 
				 "DT TF eta value", 100, -2.5, 2.5 ) ;
      dttfphivalue = dbe->book1D("DT_TF_phi_value", 
				 "DT TF phi value", 100, 0.0, 6.2832 ) ;
      dttfetapacked = dbe->book1D("DT_TF_etap", 
				 "DT TF #eta packed", 64, -0.5, 63.5 ) ;
      dttfphipacked = dbe->book1D("phip", 
				 "DT TF #phi packed", 256, -0.5, 255.5 ) ;
      dttfptvalue = dbe->book1D("DT_TF_pt_value", 
				"DT TF pt value", 160, -0.5, 159.5 ) ;
      dttfptpacked = dbe->book1D("DT_TF_pt_packed", 
				 "DT TF pt_packed", 160, -0.5, 159.5 ) ;
      dttfquality = dbe->book1D("DT_TF_quality", 
				"DT TF quality", 20, -0.5, 19.5 ) ;
      dttfchargevalue = dbe->book1D("DT_TF_charge value", 
				    "DT TF charge value", 2, -1.5, 1.5 ) ;
      dttfntrack = dbe->book1D("DT_TF_ntrack", 
			       "DT TF ntrack", 20, -0.5, 19.5 ) ;
    }  
}


void L1TDTTF::endJob(void)
{
  if(verbose_) cout << "L1TDTTF: end job...." << endl;
  LogInfo("L1TDTTF") << "analyzed " << nev_ << " events"; 

  if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

  return;
}

void L1TDTTF::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TDTTF: analyze...." << endl;


  edm::Handle<std::vector<L1MuRegionalCand> > L1DTTFTracks;  

  try {
    e.getByLabel(dttfSource_,L1DTTFTracks);
  }
  catch (...) {
    edm::LogInfo("L1TDTTF") << "can't find L1DTTFTracks with label "
			    << dttfSource_.label() ;
    return;
  }

  std::cout << "DT TF collection size: " << L1DTTFTracks->size()
   	    << std::endl;
  int ndttftrack = 0;
  for( vector<L1MuRegionalCand>::const_iterator 
	 DTTFtrackItr =  L1DTTFTracks->begin() ;
       DTTFtrackItr != L1DTTFTracks->end() ;
       ++DTTFtrackItr ) 
    {

      ndttftrack++;

      dttfetavalue->Fill(DTTFtrackItr->etaValue());     
      if (verbose_)
	{
	  std::cout << "DT TF etavalue " << DTTFtrackItr->eta_packed()  
		    << std::endl;
	}
      dttfetapacked->Fill(DTTFtrackItr->eta_packed());     
      dttfphipacked->Fill(DTTFtrackItr->phi_packed());     

      dttfphivalue->Fill(DTTFtrackItr->phiValue());     
      if (verbose_)
	{
	  std::cout << "DT TF phivalue " << DTTFtrackItr->phi_packed()  
		    << std::endl;
	}

      dttfptvalue->Fill(DTTFtrackItr->ptValue());     
      if (verbose_)
 	{
 	  std::cout << "DT TF ptvalue " << DTTFtrackItr->ptValue()  
 		    << std::endl;
 	}

      dttfptpacked->Fill(DTTFtrackItr->pt_packed());     
      if (verbose_)
	{
	  std::cout << "DT TF pt_packed " << DTTFtrackItr->pt_packed()  
		    << std::endl;
	}

      dttfquality->Fill(DTTFtrackItr->quality());     
      if (verbose_)
	{
	  std::cout << "DT TF quality " << DTTFtrackItr->quality()  
		    << std::endl;
	}

      dttfchargevalue->Fill(DTTFtrackItr->chargeValue());     
      if (verbose_)
	{
	  std::cout << "DT TF charge value " << DTTFtrackItr->chargeValue()  
		    << std::endl;
	}

    }

  dttfntrack->Fill(ndttftrack);     
  if (verbose_)
    {
      std::cout << "DT TF ntrack " << ndttftrack  
		<< std::endl;
    }
}

