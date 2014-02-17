/*
 * \file L1TRPCTF.cc
 *
 * $Date: 2011/11/15 13:31:37 $
 * $Revision: 1.33 $
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TRPCTF.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include <sstream>
using namespace std;
using namespace edm;

L1TRPCTF::L1TRPCTF(const ParameterSet& ps)
  : rpctfSource_( ps.getParameter< InputTag >("rpctfSource") ),
//    digiSource_( ps.getParameter< InputTag >("rpctfRPCDigiSource") ),
//   m_rpcDigiFine(false),
//    m_useRpcDigi(true),
   m_lastUsedBxInBxdiff(0),
   output_dir_ (ps.getUntrackedParameter<string>("output_dir") )
//    m_rpcDigiWithBX0(0),
//    m_rpcDigiWithBXnon0(0)

 {

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TRPCTF: constructor...." << endl;


  m_dbe = NULL;
  if ( ps.getUntrackedParameter<bool>("DQMStore", false) ) 
  {
    m_dbe = Service<DQMStore>().operator->();
    m_dbe->setVerbose(0);
  }

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    cout << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }


  if ( m_dbe !=NULL ) {
    m_dbe->setCurrentFolder(output_dir_);
  }



}

L1TRPCTF::~L1TRPCTF()
{
}

void L1TRPCTF::beginJob(void)
{

  nev_ = 0;
  nevRPC_ = 0;

  // get hold of back-end interface
  m_dbe = Service<DQMStore>().operator->();

  if ( m_dbe ) {
    m_dbe->setCurrentFolder(output_dir_);
    m_dbe->rmdir(output_dir_);
  }


  if ( m_dbe ) 
  {
    

    ostringstream oDir; oDir<< output_dir_ << "/CrateSynchroHistograms/";
    m_dbe->setCurrentFolder(oDir.str());
    for( unsigned int i = 0; i < 12; i++) {
      
       ostringstream o; o<<"RPCTF_crate_"<<i<<"_synchro";
       rpctfcratesynchro[i] = m_dbe->book2D(o.str(), o.str(), 5, -2.5, 2.5, 33, -16.5, 16.5);
       for (int bx = -2; bx < 3; ++bx){
          ostringstream b; b<<"BX="<<bx;
          rpctfcratesynchro[i]->setBinLabel(bx+3, b.str(),1);
       }
       rpctfcratesynchro[i]->setAxisTitle("Tower",2);
     
    }
    m_dbe->setCurrentFolder(output_dir_);
    
    rpctfetavalue[1] = m_dbe->book1D("RPCTF_eta_value_bx0", 
       "RPCTF eta value bx=0", 33, -16.5, 16.5 ) ;
    rpctfetavalue[2] = m_dbe->book1D("RPCTF_eta_value_bx+", 
       "RPCTF eta value bx>0", 33, -16.5, 16.5 ) ;
    rpctfetavalue[0] = m_dbe->book1D("RPCTF_eta_value_bx-", 
       "RPCTF eta value bx<0", 33, -16.5, 16.5 ) ;
    
    rpctfphivalue[1] = m_dbe->book1D("RPCTF_phi_value_bx0", 
       "RPCTF phi value bx=0", 144, -0.5, 143.5) ;
    rpctfphivalue[2] = m_dbe->book1D("RPCTF_phi_value_bx+", 
       "RPCTF phi value bx>0", 144, -0.5, 143.5 ) ;
    rpctfphivalue[0] = m_dbe->book1D("RPCTF_phi_value_bx-", 
       "RPCTF phi value bx<0", 144, -0.5, 143.5 ) ;
      
       
       
    rpctfptvalue[1] = m_dbe->book1D("RPCTF_pt_value_bx0", 
                                    "RPCTF pt value bx=0", 160, -0.5, 159.5 );
    rpctfptvalue[2] = m_dbe->book1D("RPCTF_pt_value_bx+", 
                                    "RPCTF pt value bx>0", 160, -0.5, 159.5 );
    rpctfptvalue[0] = m_dbe->book1D("RPCTF_pt_value_bx-", 
                                    "RPCTF pt value bx<0", 160, -0.5, 159.5 );
    
    
    rpctfchargevalue[1] = m_dbe->book1D("RPCTF_charge_value_bx0", 
                                        "RPCTF charge value bx=0", 3, -1.5, 1.5 ) ;
    rpctfchargevalue[2] = m_dbe->book1D("RPCTF_charge_value_bx+", 
                                        "RPCTF charge value bx>0", 3, -1.5, 1.5 ) ;
    rpctfchargevalue[0] = m_dbe->book1D("RPCTF_charge_value_bx-", 
                                        "RPCTF charge value bx<01", 3, -1.5, 1.5 ) ;

    rpctfquality[1] = m_dbe->book1D("RPCTF_quality", 
                                    "RPCTF quality bx=0", 6, -0.5, 5.5 ) ;
    rpctfquality[2] = m_dbe->book1D("RPCTF_quality_bx+", 
                                    "RPCTF quality bx>0", 6, -0.5, 5.5 ) ;
    rpctfquality[0] = m_dbe->book1D("RPCTF_quality_bx-", 
                                    "RPCTF quality bx<0", 6, -0.5, 5.5 ) ;

    rpctfntrack_b[1] = m_dbe->book1D("RPCTF_ntrack_brl_bx0", 
                                     "RPCTF number of tracks - barrel, bx=0", 5, -0.5, 4.5 ) ;
    rpctfntrack_b[2] = m_dbe->book1D("RPCTF_ntrack_brl_bx+", 
                                     "RPCTF number of tracks - barrel, bx>0", 5, -0.5, 4.5 ) ;
    rpctfntrack_b[0] = m_dbe->book1D("RPCTF_ntrack_brl_bx-", 
                                     "RPCTF number of tracks - barrel, bx<0", 5, -0.5, 4.5 ) ;
    
    
           
    rpctfntrack_e[1] = m_dbe->book1D("RPCTF_ntrack_fwd_bx0", 
                                     "RPCTF number of tracks - endcap, bx=0", 5, -0.5, 4.5 ) ;
    rpctfntrack_e[2] = m_dbe->book1D("RPCTF_ntrack_fwd_bx+", 
                                     "RPCTF number of tracks - endcap, bx>0", 5, -0.5, 4.5 ) ;
    rpctfntrack_e[0] = m_dbe->book1D("RPCTF_ntrack_fwd_bx-", 
                                     "RPCTF number of tracks - endcap, bx<0", 5, -0.5, 4.5 ) ;

    
                                        
       

    m_qualVsEta[1] = m_dbe->book2D("RPCTF_quality_vs_eta_bx0", 
                              "RPCTF quality vs eta, bx=0", 
                               33, -16.5, 16.5,
                               6, -0.5, 5.5); // Currently only 0...3 quals are possible
    m_qualVsEta[2] = m_dbe->book2D("RPCTF_quality_vs_eta_bx+", 
                                   "RPCTF quality vs eta, bx>0", 
                                   33, -16.5, 16.5,
                                   6, -0.5, 5.5); // Currently only 0...3 quals are possible
    m_qualVsEta[0] = m_dbe->book2D("RPCTF_quality_vs_eta_bx-", 
                                   "RPCTF quality vs eta, bx<0", 
                                   33, -16.5, 16.5,
                                   6, -0.5, 5.5); // Currently only 0...3 quals are possible
    
    
        
    m_muonsEtaPhi[1] = m_dbe->book2D("RPCTF_muons_eta_phi_bx0", 
                                  "RPCTF occupancy(eta,phi), bx=0",  
                                  33, -16.5, 16.5,
                                  144,  -0.5, 143.5);
    m_muonsEtaPhi[2] = m_dbe->book2D("RPCTF_muons_eta_phi_bx+", 
                                     "RPCTF occupancy(eta,phi), bx>0",  
                                     33, -16.5, 16.5,
                                     144,  -0.5, 143.5);
    m_muonsEtaPhi[0] = m_dbe->book2D("RPCTF_muons_eta_phi_bx-", 
                                     "RPCTF occupancy(eta,phi), bx<0",  
                                     33, -16.5, 16.5,
                                     144,  -0.5, 143.5);

    rpctfbx = m_dbe->book1D("RPCTF_bx", 
                            "RPCTF bx distribiution", 7, -3.5, 3.5 );
    
    //axis labels
    for (int l = 0; l<3; ++l){
      m_muonsEtaPhi[l]->setAxisTitle("tower",1);
      m_qualVsEta[l]->setAxisTitle("tower");
      rpctfetavalue[l]->setAxisTitle("tower");
      
      m_muonsEtaPhi[l]->setAxisTitle("phi",2);
      rpctfphivalue[l]->setAxisTitle("phi");
    }
    
    // set phi bin labels
    for (int i = 0; i < 12 ; ++i ){
       //float lPhi  = (30./360)*i*2*3.14;
      int lPhi  = 30*i;
      int lBin = int((30./360)*i*144)+1;
      std::stringstream ss;
      ss << "phi=" <<lPhi;
      for (int l = 0; l<3; ++l){
        rpctfphivalue[l]->setBinLabel(lBin,ss.str());
        m_muonsEtaPhi[l]->setBinLabel(lBin,ss.str(), 2);
      }
    }

    /*
    // set TC numbers on phi axis
    for (int tc = 0; tc < 12 ; ++tc ){
      int lBin  = (tc*12+3+1)%144;
      std::stringstream ss;
      ss << "TC" <<tc;
      for (int l = 0; l<3; ++l){
        rpctfphivalue[l]->setBinLabel(lBin,ss.str());
        m_muonsEtaPhi[l]->setBinLabel(lBin,ss.str(), 2);
      }
  }*/

        
    // set eta bin labels
    for (int i = -16; i < 17 ; ++i ){
      std::stringstream ss;
      ss << i;
      for (int l = 0; l<3; ++l){
        rpctfetavalue[l]->setBinLabel(i+17, ss.str());
        m_muonsEtaPhi[l]->setBinLabel(i+17, ss.str(), 1);
        m_qualVsEta[l]->setBinLabel(i+17, ss.str());
      }
    }

    
                              
    m_bxDiff = m_dbe->book1D("RPCTF_bx_diff",
			      "RPCTrigger - bx difference", 12000, -.5, 11999.5); 
   


  }   // if (m_dbe)
}

void L1TRPCTF::endRun(const edm::Run & r, const edm::EventSetup & c){
  


     // fixme, norm iteration would be better
     while (m_globBX.begin() !=  m_globBX.end() ) {
        long long int  diff = *m_globBX.begin()-m_lastUsedBxInBxdiff; // first entry will go to overflow bin, ignore
        m_bxDiff->Fill(diff);
        m_lastUsedBxInBxdiff = *m_globBX.begin();
        m_globBX.erase(m_globBX.begin());

     }

     
}


void L1TRPCTF::endJob(void)
{
  
  if(verbose_) cout << "L1TRPCTF: end job...." << endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

  if ( outputFile_.size() != 0  && m_dbe ) m_dbe->save(outputFile_);
    
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

  std::vector<int> nrpctftrack_b(3,0);
  std::vector<int> nrpctftrack_e(3,0);

  vector<L1TRPCTF::BxDelays> all_bxdelays;


  for( RRItr = gmt_records.begin() ;
       RRItr != gmt_records.end() ;
       RRItr++ ) 
  {
    
   if (verbose_) cout << "Readout Record " << RRItr->getBxInEvent() << endl;
   
   vector<vector<L1MuRegionalCand> > brlAndFwdCands;
   brlAndFwdCands.push_back(RRItr->getBrlRPCCands());
   brlAndFwdCands.push_back(RRItr->getFwdRPCCands());
  
   int beIndex = 0;
   vector<vector<L1MuRegionalCand> >::iterator RPCTFCands = brlAndFwdCands.begin();
   for(; RPCTFCands!= brlAndFwdCands.end(); ++RPCTFCands)
   {
      
      for( vector<L1MuRegionalCand>::const_iterator 
          ECItr = RPCTFCands->begin() ;
          ECItr != RPCTFCands->end() ;
          ++ECItr ) 
      {
  
        int bxindex = 1 ; // bx == 0
        if (ECItr->bx() > 0) bxindex = 2;
        if (ECItr->bx() < 0) bxindex = 0;
        
        if (!ECItr->empty()) {
          
          
          if (beIndex == 0) ++nrpctftrack_b[bxindex];
          if (beIndex == 1) ++nrpctftrack_e[bxindex];
    
          if (verbose_) cout << "RPCTFCand bx " << ECItr->bx() << endl;
          
          int tower = ECItr->eta_packed();
          if (tower > 16) {
            tower = - ( (~tower & 63) + 1);
          }

          rpctfbx->Fill(ECItr->bx());
    
          rpctfetavalue[bxindex]->Fill(tower);
          if (verbose_) cout << "\tRPCTFCand eta value " << ECItr->etaValue() << endl;
  
          rpctfphivalue[bxindex]->Fill(ECItr->phi_packed());
          if (verbose_) cout << "\tRPCTFCand phi value " << ECItr->phiValue() << endl;
    
          rpctfptvalue[bxindex]->Fill(ECItr->ptValue());
          if (verbose_) cout << "\tRPCTFCand pt value " << ECItr->ptValue()<< endl;
    
          rpctfchargevalue[bxindex]->Fill(ECItr->chargeValue());
          if (verbose_) cout << "\tRPCTFCand charge value " << ECItr->chargeValue() << endl;
    
          rpctfquality[bxindex]->Fill(ECItr->quality());
          if (verbose_) cout << "\tRPCTFCand quality " << ECItr->quality() << endl;
          

          m_qualVsEta[bxindex]->Fill(tower, ECItr->quality());
          m_muonsEtaPhi[bxindex]->Fill(tower, ECItr->phi_packed());

          BxDelays bx_del;
          bx_del.bx = ECItr->bx();
          bx_del.eta_t = tower;
          bx_del.phi_p = ECItr->phi_packed();
          all_bxdelays.push_back(bx_del);
          
        } // if !empty
      } // end candidates iteration
      ++beIndex;
   } // end brl/endcap iteration
  } // end GMT records iteration

  for (int bxI = 0; bxI < 3; ++bxI){
    rpctfntrack_b[bxI]->Fill(nrpctftrack_b[bxI]);
    rpctfntrack_e[bxI]->Fill(nrpctftrack_e[bxI]);
  }
  
  
   for(unsigned int i = 0; i < all_bxdelays.size(); i++) {

     int sector= ((all_bxdelays[i].phi_p+ 142)%144)/12;
     if (sector>11 || sector < 0) continue;
     int eta_tower = all_bxdelays[i].eta_t;
     for(unsigned int j = 0; j < all_bxdelays.size(); j++) {
       if(i == j) continue;
       int sector2= ((all_bxdelays[j].phi_p + 142)%144)/12;
 
       int distance_cut = 1;
       int distance = ((sector+12)-sector2)%12;
       distance = min(distance, 11-distance);
       if(distance<distance_cut) continue;
 
       int bxDiff = all_bxdelays[i].bx-all_bxdelays[j].bx;
       rpctfcratesynchro[sector]->Fill(bxDiff,eta_tower);
     }

  }
  
  
  	
}

void L1TRPCTF::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                                    const edm::EventSetup& context)
{
//    m_rpcDigiWithBX0=0;
//    m_rpcDigiWithBXnon0=0;
//    m_bxs.clear();
//    m_useRpcDigi = true;

                          
}


void L1TRPCTF::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                        const edm::EventSetup& c)
{

}

