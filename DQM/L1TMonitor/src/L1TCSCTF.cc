/*
 * \file L1TCSCTF.cc
 *
 * $Date: 2008/03/14 20:35:46 $
 * $Revision: 1.15 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TCSCTF.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
using namespace edm;

L1TCSCTF::L1TCSCTF(const ParameterSet& ps)
  : csctfSource_( ps.getParameter< InputTag >("csctfSource") )
 {

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TCSCTF: constructor...." << endl;


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

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if ( dbe !=NULL ) {
    dbe->setCurrentFolder("L1T/L1TCSCTF");
  }


}

L1TCSCTF::~L1TCSCTF()
{
}

void L1TCSCTF::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1TCSCTF");
    dbe->rmdir("L1T/L1TCSCTF");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1T/L1TCSCTF");
    
    csctfetavalue[1] = dbe->book1D("CSCTF_eta_value", 
       "CSCTF eta value", 100, -2.5, 2.5 ) ;
    csctfetavalue[2] = dbe->book1D("CSCTF_eta_value_+1", 
       "CSCTF eta value bx +1", 100, -2.5, 2.5 ) ;
    csctfetavalue[0] = dbe->book1D("CSCTF_eta_value_-1", 
       "CSCTF eta value bx -1", 100, -2.5, 2.5 ) ;
    csctfphivalue[1] = dbe->book1D("CSCTF_phi_value", 
       "CSCTF phi value", 100, 0.0, 6.2832 ) ;
    csctfphivalue[2] = dbe->book1D("CSCTF_phi_value_+1", 
       "CSCTF phi value bx +1", 100, 0.0, 6.2832 ) ;
    csctfphivalue[0] = dbe->book1D("CSCTF_phi_value_-1", 
       "CSCTF phi value bx -1", 100, 0.0, 6.2832 ) ;
    csctfptvalue[1] = dbe->book1D("CSCTF_pt_value", 
       "CSCTF pt value", 160, -0.5, 159.5 ) ;
    csctfptvalue[2] = dbe->book1D("CSCTF_pt_value_+1", 
       "CSCTF pt value bx +1", 160, -0.5, 159.5 ) ;
    csctfptvalue[0] = dbe->book1D("CSCTF_pt_value_-1", 
       "CSCTF pt value bx -1", 160, -0.5, 159.5 ) ;
    csctfchargevalue[1] = dbe->book1D("CSCTF_charge_value", 
       "CSCTF charge value", 3, -1.5, 1.5 ) ;
    csctfchargevalue[2] = dbe->book1D("CSCTF_charge_value_+1", 
       "CSCTF charge value bx +1", 3, -1.5, 1.5 ) ;
    csctfchargevalue[0] = dbe->book1D("CSCTF_charge_value_-1", 
       "CSCTF charge value bx -1", 3, -1.5, 1.5 ) ;
    csctfquality[1] = dbe->book1D("CSCTF_quality", 
       "CSCTF quality", 20, -0.5, 19.5 ) ;
    csctfquality[2] = dbe->book1D("CSCTF_quality_+1", 
       "CSCTF quality bx +1", 20, -0.5, 19.5 ) ;
    csctfquality[0] = dbe->book1D("CSCTF_quality_-1", 
       "CSCTF quality bx -1", 20, -0.5, 19.5 ) ;
    csctfntrack = dbe->book1D("CSCTF_ntrack", 
       "CSCTF ntrack", 20, -0.5, 19.5 ) ;
    csctfbx = dbe->book1D("CSCTF_bx", 
       "CSCTF bx", 3, -1.5, 1.5 ) ;
  }  
}


void L1TCSCTF::endJob(void)
{
  if(verbose_) cout << "L1TCSCTF: end job...." << endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TCSCTF::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TCSCTF: analyze...." << endl;


  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  e.getByLabel(csctfSource_,pCollection);
  
  if (!pCollection.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1MuGMTReadoutCollection with label "
			       << csctfSource_.label() ;
    return;
  }

  L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
  vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  vector<L1MuGMTReadoutRecord>::const_iterator RRItr;

  int ncsctftrack = 0;
  for( RRItr = gmt_records.begin() ;
       RRItr != gmt_records.end() ;
       RRItr++ ) 
  {

    if (verbose_)
    {
     cout << "Readout Record " << RRItr->getBxInEvent()
   	    << endl;
   }
 
   vector<L1MuRegionalCand> CSCTFCands = RRItr->getCSCCands();
 

   if (verbose_) 
    {
     cout << "CSCTFCands " << CSCTFCands.size()
   	    << endl;
    }

    for( vector<L1MuRegionalCand>::const_iterator 
         ECItr = CSCTFCands.begin() ;
         ECItr != CSCTFCands.end() ;
         ++ECItr ) 
    {

      int bxindex = ECItr->bx() + 1;

      if (ECItr->quality() > 0 )
     {
      ncsctftrack++;

      if (verbose_)
	{  
     cout << "CSCTFCand bx " << ECItr->bx()
   	    << endl;
	}
     csctfbx->Fill(ECItr->bx());

      csctfetavalue[bxindex]->Fill(ECItr->etaValue());
      if (verbose_)
	{     
     cout << "\tCSCTFCand eta value " << ECItr->etaValue()
   	    << endl;
	}

      csctfphivalue[bxindex]->Fill(ECItr->phiValue());
      if (verbose_)
	{     
     cout << "\tCSCTFCand phi value " << ECItr->phiValue()
   	    << endl;
	}

      csctfptvalue[bxindex]->Fill(ECItr->ptValue());
      if (verbose_)
	{     
     cout << "\tCSCTFCand pt value " << ECItr->ptValue()
   	    << endl;
	}


      csctfchargevalue[bxindex]->Fill(ECItr->chargeValue());
      if (verbose_)
	{     
     cout << "\tCSCTFCand charge value " << ECItr->chargeValue()
   	    << endl;
	}

      csctfquality[bxindex]->Fill(ECItr->quality());
      if (verbose_)
	{     
     cout << "\tCSCTFCand quality " << ECItr->quality()
   	    << endl;
	}

     }
    }


  }

      csctfntrack->Fill(ncsctftrack);
      if (verbose_)
	{     
     cout << "\tCSCTFCand ntrack " << ncsctftrack
   	    << endl;
	}
}

