/*
 * \file L1TDTTF.cc
 *
 * $Date: 2007/12/21 17:41:20 $
 * $Revision: 1.8 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TDTTF.h"
#include "DQMServices/Core/interface/DQMStore.h"

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
  if ( ps.getUntrackedParameter<bool>("DQMStore", false) ) 
  {
    dbe = Service<DQMStore>().operator->();
    dbe->setVerbose(0);
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
    dbe->setCurrentFolder("L1T/L1TDTTF");
  }


}

L1TDTTF::~L1TDTTF()
{
}

void L1TDTTF::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1TDTTF");
    dbe->rmdir("L1T/L1TDTTF");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1T/L1TDTTF");
    
    dttfetavalue[1] = dbe->book1D("DTTF_eta_value", 
       "DTTF eta value", 100, -2.5, 2.5 ) ;
    dttfetavalue[2] = dbe->book1D("DTTF_eta_value_+1", 
       "DTTF eta value bx +1", 100, -2.5, 2.5 ) ;
    dttfetavalue[0] = dbe->book1D("DTTF_eta_value_-1", 
       "DTTF eta value bx -1", 100, -2.5, 2.5 ) ;
    dttfphivalue[1] = dbe->book1D("DTTF_phi_value", 
       "DTTF phi value", 100, 0.0, 6.2832 ) ;
    dttfphivalue[2] = dbe->book1D("DTTF_phi_value_+1", 
       "DTTF phi value bx +1", 100, 0.0, 6.2832 ) ;
    dttfphivalue[0] = dbe->book1D("DTTF_phi_value_-1", 
       "DTTF phi value bx -1", 100, 0.0, 6.2832 ) ;
    dttfptvalue[1] = dbe->book1D("DTTF_pt_value", 
       "DTTF pt value", 160, -0.5, 159.5 ) ;
    dttfptvalue[2] = dbe->book1D("DTTF_pt_value_+1", 
       "DTTF pt value bx +1", 160, -0.5, 159.5 ) ;
    dttfptvalue[0] = dbe->book1D("DTTF_pt_value_-1", 
       "DTTF pt value bx -1", 160, -0.5, 159.5 ) ;
    dttfchargevalue[1] = dbe->book1D("DTTF_charge_value", 
       "DTTF charge value", 3, -1.5, 1.5 ) ;
    dttfchargevalue[2] = dbe->book1D("DTTF_charge_value_+1", 
       "DTTF charge value bx +1", 3, -1.5, 1.5 ) ;
    dttfchargevalue[0] = dbe->book1D("DTTF_charge_value_-1", 
       "DTTF charge value bx -1", 3, -1.5, 1.5 ) ;
    dttfquality[1] = dbe->book1D("DTTF_quality", 
       "DTTF quality", 20, -0.5, 19.5 ) ;
    dttfquality[2] = dbe->book1D("DTTF_quality_+1", 
       "DTTF quality bx +1", 20, -0.5, 19.5 ) ;
    dttfquality[0] = dbe->book1D("DTTF_quality_-1", 
       "DTTF quality bx -1", 20, -0.5, 19.5 ) ;
    dttfntrack = dbe->book1D("DTTF_ntrack", 
       "DTTF ntrack", 20, -0.5, 19.5 ) ;
    dttfbx = dbe->book1D("DTTF_bx", 
       "DTTF bx", 3, -1.5, 1.5 ) ;
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


  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  e.getByLabel(dttfSource_,pCollection);
  
  if (!pCollection.isValid()) {
    edm::LogInfo("L1TDTTF") << "can't find L1MuGMTReadoutCollection with label "
			       << dttfSource_.label() ;
    return;
  }

  L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
  vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  vector<L1MuGMTReadoutRecord>::const_iterator RRItr;

  int ndttftrack = 0;
  for( RRItr = gmt_records.begin() ;
       RRItr != gmt_records.end() ;
       RRItr++ ) 
  {

    if (verbose_)
    {
     cout << "Readout Record " << RRItr->getBxInEvent()
   	    << endl;
   }
 
   vector<L1MuRegionalCand> DTTFCands = RRItr->getDTBXCands();
 

   if (verbose_) 
    {
     cout << "DTTFCands " << DTTFCands.size()
   	    << endl;
    }

    for( vector<L1MuRegionalCand>::const_iterator 
         ECItr = DTTFCands.begin() ;
         ECItr != DTTFCands.end() ;
         ++ECItr ) 
    {

      int bxindex = ECItr->bx() + 1;

      if (ECItr->quality() > 0 )
     {
      ndttftrack++;

      if (verbose_)
	{  
     cout << "DTTFCand bx " << ECItr->bx()
   	    << endl;
	}
     dttfbx->Fill(ECItr->bx());

      dttfetavalue[bxindex]->Fill(ECItr->etaValue());
      if (verbose_)
	{     
     cout << "\tDTTFCand eta value " << ECItr->etaValue()
   	    << endl;
	}

      dttfphivalue[bxindex]->Fill(ECItr->phiValue());
      if (verbose_)
	{     
     cout << "\tDTTFCand phi value " << ECItr->phiValue()
   	    << endl;
	}

      dttfptvalue[bxindex]->Fill(ECItr->ptValue());
      if (verbose_)
	{     
     cout << "\tDTTFCand pt value " << ECItr->ptValue()
   	    << endl;
	}


      dttfchargevalue[bxindex]->Fill(ECItr->chargeValue());
      if (verbose_)
	{     
     cout << "\tDTTFCand charge value " << ECItr->chargeValue()
   	    << endl;
	}

      dttfquality[bxindex]->Fill(ECItr->quality());
      if (verbose_)
	{     
     cout << "\tDTTFCand quality " << ECItr->quality()
   	    << endl;
	}

     }
    }


  }

      dttfntrack->Fill(ndttftrack);
      if (verbose_)
	{     
     cout << "\tDTTFCand ntrack " << ndttftrack
   	    << endl;
	}
}

