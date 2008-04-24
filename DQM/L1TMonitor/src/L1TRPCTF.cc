/*
 * \file L1TRPCTF.cc
 *
 * $Date: 2008/04/11 15:26:10 $
 * $Revision: 1.14 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TRPCTF.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
using namespace edm;

L1TRPCTF::L1TRPCTF(const ParameterSet& ps)
  : rpctfSource_( ps.getParameter< InputTag >("rpctfSource") ),
                  m_ntracks(0)
 {

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TRPCTF: constructor...." << endl;


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
    dbe->setCurrentFolder("L1T/L1TRPCTF");
  }


}

L1TRPCTF::~L1TRPCTF()
{
}

void L1TRPCTF::beginJob(const EventSetup& c)
{

  nev_ = 0;

  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1TRPCTF");
    dbe->rmdir("L1T/L1TRPCTF");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1T/L1TRPCTF");
    
    rpctfetavalue[1] = dbe->book1D("RPCTF_eta_value", 
       "RPCTF eta value", 100, -2.5, 2.5 ) ;
    rpctfetavalue[2] = dbe->book1D("RPCTF_eta_value_+1", 
       "RPCTF eta value bx +1", 100, -2.5, 2.5 ) ;
    rpctfetavalue[0] = dbe->book1D("RPCTF_eta_value_-1", 
       "RPCTF eta value bx -1", 100, -2.5, 2.5 ) ;
    rpctfphivalue[1] = dbe->book1D("RPCTF_phi_value", 
       "RPCTF phi value", 144, 0.0, 6.2832 ) ;
    rpctfphivalue[2] = dbe->book1D("RPCTF_phi_value_+1", 
       "RPCTF phi value bx +1", 144, 0.0, 6.2832 ) ;
    rpctfphivalue[0] = dbe->book1D("RPCTF_phi_value_-1", 
       "RPCTF phi value bx -1", 144, 0.0, 6.2832 ) ;
    rpctfptvalue[1] = dbe->book1D("RPCTF_pt_value", 
       "RPCTF pt value", 160, -0.5, 159.5 ) ;
    rpctfptvalue[2] = dbe->book1D("RPCTF_pt_value_+1", 
       "RPCTF pt value bx +1", 160, -0.5, 159.5 ) ;
    rpctfptvalue[0] = dbe->book1D("RPCTF_pt_value_-1", 
       "RPCTF pt value bx -1", 160, -0.5, 159.5 ) ;
    rpctfchargevalue[1] = dbe->book1D("RPCTF_charge_value", 
       "RPCTF charge value", 3, -1.5, 1.5 ) ;
    rpctfchargevalue[2] = dbe->book1D("RPCTF_charge_value_+1", 
       "RPCTF charge value bx +1", 3, -1.5, 1.5 ) ;
    rpctfchargevalue[0] = dbe->book1D("RPCTF_charge_value_-1", 
       "RPCTF charge value bx -1", 3, -1.5, 1.5 ) ;
    rpctfquality[1] = dbe->book1D("RPCTF_quality", 
       "RPCTF quality", 20, -0.5, 19.5 ) ;
    rpctfquality[2] = dbe->book1D("RPCTF_quality_+1", 
       "RPCTF quality bx +1", 20, -0.5, 19.5 ) ;
    rpctfquality[0] = dbe->book1D("RPCTF_quality_-1", 
       "RPCTF quality bx -1", 20, -0.5, 19.5 ) ;
    rpctfntrack = dbe->book1D("RPCTF_ntrack", 
       "RPCTF ntrack", 20, -0.5, 19.5 ) ;
    rpctfbx = dbe->book1D("RPCTF_bx", 
       "RPCTF bx", 3, -1.5, 1.5 ) ;
    
    m_qualVsEta = dbe->book2D("RPCTF_quality_vs_eta_value", 
                              "RPCTF quality vs eta", 
                              100, -2.5, 2.5,
                               10, -0.5, 9.5);
    
    m_muonsEtaPhi = dbe->book2D("RPCTF_muons_eta_phipacked", 
                                "RPCTF muons(eta,phipacked)",  
                                100, -2.5, 2.5,
                                144,  -0.5, 143.5);
    
    m_phipacked = dbe->book1D("RPCTF_phi_valuepacked", 
                           "RPCTF phi valuepacked", 144, -0.5, 143.5 ) ;

    m_phipackednorm = dbe->book1D("RPCTF_phi_valuepacked_norm",
                                   "RPCTF phi valuepacked", 144, -0.5, 143.5 ) ;
    
        
  }  
}


void L1TRPCTF::endJob(void)
{
  if(verbose_) cout << "L1TRPCTF: end job...." << endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TRPCTF::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TRPCTF: analyze...." << endl;

  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  e.getByLabel(rpctfSource_,pCollection);
  
  if (!pCollection.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1MuGMTReadoutCollection with label "
			       << rpctfSource_.label() ;
    return;
  }

  L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
  vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  vector<L1MuGMTReadoutRecord>::const_iterator RRItr;

  int nrpctftrack = 0;
  for( RRItr = gmt_records.begin() ;
       RRItr != gmt_records.end() ;
       RRItr++ ) 
  {
    
   if (verbose_) cout << "Readout Record " << RRItr->getBxInEvent() << endl;
   
   vector<vector<L1MuRegionalCand> > brlAndFwdCands;
   brlAndFwdCands.push_back(RRItr->getBrlRPCCands());
   brlAndFwdCands.push_back(RRItr->getFwdRPCCands());
   
   //if (verbose_) cout << "RPCTFCands " << RPCTFCands.size() << endl;
   
   vector<vector<L1MuRegionalCand> >::iterator RPCTFCands = brlAndFwdCands.begin();
   for(; RPCTFCands!= brlAndFwdCands.end(); ++RPCTFCands)
   {
      for( vector<L1MuRegionalCand>::const_iterator 
          ECItr = RPCTFCands->begin() ;
          ECItr != RPCTFCands->end() ;
          ++ECItr ) 
      {
  
        int bxindex = ECItr->bx() + 1;
        
        if (!ECItr->empty()) {
          
          nrpctftrack++;
    
          if (verbose_) cout << "RPCTFCand bx " << ECItr->bx() << endl;
          rpctfbx->Fill(ECItr->bx());
    
          rpctfetavalue[bxindex]->Fill(ECItr->etaValue());
          if (verbose_) cout << "\tRPCTFCand eta value " << ECItr->etaValue() << endl;
  
          rpctfphivalue[bxindex]->Fill(ECItr->phiValue());
          if (verbose_) cout << "\tRPCTFCand phi value " << ECItr->phiValue() << endl;
    
          rpctfptvalue[bxindex]->Fill(ECItr->ptValue());
          if (verbose_) cout << "\tRPCTFCand pt value " << ECItr->ptValue()<< endl;
    
          rpctfchargevalue[bxindex]->Fill(ECItr->chargeValue());
          if (verbose_) cout << "\tRPCTFCand charge value " << ECItr->chargeValue() << endl;
    
          rpctfquality[bxindex]->Fill(ECItr->quality());
          if (verbose_) cout << "\tRPCTFCand quality " << ECItr->quality() << endl;
          
          m_qualVsEta->Fill(ECItr->etaValue(), ECItr->quality());
          
          m_muonsEtaPhi->Fill(ECItr->etaValue(), ECItr->phi_packed());
          m_phipacked->Fill(ECItr->phi_packed());
          
        } // if !empty
      } // end candidates iteration
   } // end brl/endcap iteration
  } // end GMT records iteration

  rpctfntrack->Fill(nrpctftrack);
  
  m_ntracks += nrpctftrack;
  
  
  if (verbose_) cout << "\tRPCTFCand ntrack " << nrpctftrack << endl;
	
}

// clear normalization values
void L1TRPCTF::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                                    const edm::EventSetup& context)
{
   m_ntracks = 0;
                          
}

// Fill normalized histograms.  
void L1TRPCTF::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                        const edm::EventSetup& c)
{
   
   float ntracks = m_ntracks; 
   // todo check if m_phipackednorm and RPCTF_phi_valuepacked have same number of bins
   if (ntracks > 0.5) {
      for (int bin = 0; bin <= m_phipackednorm->getNbinsX(); ++bin) {
         m_phipackednorm->setBinContent(bin, m_phipacked->getBinContent(bin)/ntracks);
      }
   }
   
   
}

