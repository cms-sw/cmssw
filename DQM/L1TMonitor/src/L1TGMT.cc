/*
 * \file L1TGMT.cc
 *
 * $Date: 2007/07/26 08:56:53 $
 * $Revision: 1.6 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TGMT.h"

using namespace std;
using namespace edm;

L1TGMT::L1TGMT(const ParameterSet& ps)
  : gmtSource_( ps.getParameter< InputTag >("gmtSource") )
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
    
    gmtetavalue[1] = dbe->book1D("GMT_eta_value", 
       "GMT eta value", 100, -2.5, 2.5 ) ;
    gmtetavalue[2] = dbe->book1D("GMT_eta_value_+1", 
       "GMT eta value bx +1", 100, -2.5, 2.5 ) ;
    gmtetavalue[0] = dbe->book1D("GMT_eta_value_-1", 
       "GMT eta value bx -1", 100, -2.5, 2.5 ) ;
    gmtphivalue[1] = dbe->book1D("GMT_phi_value", 
       "GMT phi value", 100, 0.0, 6.2832 ) ;
    gmtphivalue[2] = dbe->book1D("GMT_phi_value_+1", 
       "GMT phi value bx +1", 100, 0.0, 6.2832 ) ;
    gmtphivalue[0] = dbe->book1D("GMT_phi_value_-1", 
       "GMT phi value bx -1", 100, 0.0, 6.2832 ) ;
    gmtptvalue[1] = dbe->book1D("GMT_pt_value", 
       "GMT pt value", 160, -0.5, 159.5 ) ;
    gmtptvalue[2] = dbe->book1D("GMT_pt_value_+1", 
       "GMT pt value bx +1", 160, -0.5, 159.5 ) ;
    gmtptvalue[0] = dbe->book1D("GMT_pt_value_-1", 
       "GMT pt value bx -1", 160, -0.5, 159.5 ) ;
    gmtquality[1] = dbe->book1D("GMT_quality", 
       "GMT quality", 20, -0.5, 19.5 ) ;
    gmtquality[2] = dbe->book1D("GMT_quality_+1", 
       "GMT quality bx +1", 20, -0.5, 19.5 ) ;
    gmtquality[0] = dbe->book1D("GMT_quality_-1", 
       "GMT quality bx -1", 20, -0.5, 19.5 ) ;
    gmtcharge[1] = dbe->book1D("GMT_charge", 
       "GMT charge", 2, -1.5, 1.5 ) ;
    gmtcharge[2] = dbe->book1D("GMT_charge_+1", 
       "GMT charge bx +1", 2, -1.5, 1.5 ) ;
    gmtcharge[0] = dbe->book1D("GMT_charge_-1", 
       "GMT charge bx -1", 2, -1.5, 1.5 ) ;
    gmtntrack = dbe->book1D("GMT_ntrack", 
       "GMT ntrack", 20, -0.5, 19.5 ) ;
    gmtbx = dbe->book1D("GMT_bx", 
       "GMT bx", 3, -1.5, 1.5 ) ;
    gmtbxrr = dbe->book1D("GMT_bxrr", 
       "GMT bxrr", 3, -1.5, 1.5 ) ;
    

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


  try {
  e.getByLabel(gmtSource_,pCollection);
  }
  catch (...) {
    edm::LogInfo("L1TGMT") << "can't find L1MuGMTReadoutCollection with label "
			       << gmtSource_.label() ;
    return;
  }

  L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
  vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  vector<L1MuGMTReadoutRecord>::const_iterator RRItr;

  int ngmttrack = 0;
  for( RRItr = gmt_records.begin() ;
       RRItr != gmt_records.end() ;
       RRItr++ ) 
  {

    if (verbose_)
    {
     cout << "Readout Record " << RRItr->getBxInEvent()
   	    << endl;
   }
   gmtbxrr->Fill(RRItr->getBxInEvent());

   vector<L1MuGMTExtendedCand> GMTBrlCands = RRItr->getGMTBrlCands();
 

   if (verbose_) 
    {
     cout << "GMTCands " << GMTBrlCands.size()
   	    << endl;
    }

    for( vector<L1MuGMTExtendedCand>::const_iterator 
         ECItr = GMTBrlCands.begin() ;
         ECItr != GMTBrlCands.end() ;
         ++ECItr ) 
    {

      int bxindex = ECItr->bx() + 1;
      ngmttrack++;

      if (verbose_)
	{  
     cout << "GMTCand name " << ECItr->name()
   	    << endl;
	}

      if (verbose_)
	{  
     cout << "GMTCand bx " << ECItr->bx()
   	    << endl;
	}
     gmtbx->Fill(ECItr->bx());

      gmtetavalue[bxindex]->Fill(ECItr->etaValue());
      if (verbose_)
	{     
     cout << "\tGMTCand eta value " << ECItr->etaValue()
   	    << endl;
	}

      gmtphivalue[bxindex]->Fill(ECItr->phiValue());
      if (verbose_)
	{     
     cout << "\tGMTCand phi value " << ECItr->phiValue()
   	    << endl;
	}

      gmtptvalue[bxindex]->Fill(ECItr->ptValue());
      if (verbose_)
	{     
     cout << "\tGMTCand pt value " << ECItr->ptValue()
   	    << endl;
	}

      gmtquality[bxindex]->Fill(ECItr->quality());
      if (verbose_)
	{     
     cout << "\tGMTCand quality " << ECItr->quality()
   	    << endl;
	}

      gmtcharge[bxindex]->Fill(ECItr->charge());
      if (verbose_)
	{     
     cout << "\tGMTCand charge " << ECItr->charge()
   	    << endl;
	}

    }


   vector<L1MuGMTExtendedCand> GMTFwdCands = RRItr->getGMTFwdCands();
 

   if (verbose_) 
    {
     cout << "GMTCands " << GMTFwdCands.size()
   	    << endl;
    }

    for( vector<L1MuGMTExtendedCand>::const_iterator 
         ECItr = GMTFwdCands.begin() ;
         ECItr != GMTFwdCands.end() ;
         ++ECItr ) 
    {
      
      int bxindex = ECItr->bx() + 1;
      ngmttrack++;

      if (verbose_)
	{  
     cout << "GMTCand name " << ECItr->name()
   	    << endl;
	}


      if (verbose_)
	{  
     cout << "GMTCand bx " << ECItr->bx()
   	    << endl;
	}
     gmtbx->Fill(ECItr->bx());

      gmtetavalue[bxindex]->Fill(ECItr->etaValue());
      if (verbose_)
	{     
     cout << "\tGMTCand eta value " << ECItr->etaValue()
   	    << endl;
	}

      gmtphivalue[bxindex]->Fill(ECItr->phiValue());
      if (verbose_)
	{     
     cout << "\tGMTCand phi value " << ECItr->phiValue()
   	    << endl;
	}

      gmtptvalue[bxindex]->Fill(ECItr->ptValue());
      if (verbose_)
	{     
     cout << "\tGMTCand pt value " << ECItr->ptValue()
   	    << endl;
	}

      gmtquality[bxindex]->Fill(ECItr->quality());
      if (verbose_)
	{     
     cout << "\tGMTCand quality " << ECItr->quality()
   	    << endl;
	}

      gmtcharge[bxindex]->Fill(ECItr->charge());
      if (verbose_)
	{     
     cout << "\tGMTCand charge " << ECItr->charge()
   	    << endl;
	}

    }
  }

      gmtntrack->Fill(ngmttrack);
      if (verbose_)
	{     
     cout << "\tGMTCand ntrack " << ngmttrack
   	    << endl;
	}
}

