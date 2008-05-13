/*
 * \file L1TGMT.cc
 *
 * $Date: 2008/05/07 12:05:55 $
 * $Revision: 1.18 $
 * \author J. Berryhill, I. Mikulec
 *
 */

#include "DQM/L1TMonitor/interface/L1TGMT.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

using namespace std;
using namespace edm;

const double L1TGMT::piconv_ = 180. / acos(-1.);

L1TGMT::L1TGMT(const ParameterSet& ps)
  : gmtSource_( ps.getParameter< InputTag >("gmtSource") )
 {

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) cout << "L1TGMT: constructor...." << endl;


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
    dbe->setCurrentFolder("L1T/L1TGMT");
  }

}

L1TGMT::~L1TGMT()
{
}

void L1TGMT::beginJob(const EventSetup& c)
{

  std::string subs[5] = { "DTTF", "RPCb", "CSCTF", "RPCf", "GMT" };

  nev_ = 0;
  evnum_old_ = -1;
  bxnum_old_ = -1;

  edm::ESHandle< L1MuTriggerScales > trigscales_h;
  c.get< L1MuTriggerScalesRcd >().get( trigscales_h );
  const L1MuTriggerScales* scales = trigscales_h.product();

  edm::ESHandle< L1MuTriggerPtScale > trigptscale_h;
  c.get< L1MuTriggerPtScaleRcd >().get( trigptscale_h );
  const L1MuTriggerPtScale* scalept = trigptscale_h.product();  

  // get hold of back-end interface
  DQMStore* dbe = 0;
  dbe = Service<DQMStore>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("L1T/L1TGMT");
    dbe->rmdir("L1T/L1TGMT");
  }


  if ( dbe ) 
  {
    dbe->setCurrentFolder("L1T/L1TGMT");
    
    int nphi=144; double phimin=  0.; double phimax=360.;
    int nqty=  8; double qtymin=-0.5; double qtymax=7.5;

    float phiscale[145];
    {
      for(int j=0; j<145; j++) {
        phiscale[j] = j*2.5 ;
      }
    }

    float qscale[9];
    {
      for(int j=0; j<9; j++) {
        qscale[j] = -0.5 + j;
      }
    } 
    
    float ptscale[32];
    {
      int i=0;
      for(int j=1; j<32; j++,i++) {
        ptscale[i] = scalept->getPtScale()->getLowEdge(j);
      }
      ptscale[31]=ptscale[30]+10.;
    }
      
    float etascale[5][66];
    int netascale[5];
    // DTTF eta scale
    {
      int i=0;
      for(int j=32; j<=63; j++,i++) {
        etascale[DTTF][i] = scales->getRegionalEtaScale(DTTF)->getLowEdge(j);
      }
      for(int j=0; j<=31; j++,i++) {
        etascale[DTTF][i] = scales->getRegionalEtaScale(DTTF)->getLowEdge(j);
      }
      etascale[DTTF][64] = scales->getRegionalEtaScale(DTTF)->getScaleMax();
      netascale[DTTF]=64;
    }
    // RPCb etascale
    {
      int i=0;
      for(int j=48; j<=63; j++,i++) {
        etascale[RPCb][i] = scales->getRegionalEtaScale(RPCb)->getLowEdge(j);
      }
      for(int j=0; j<=16; j++,i++) {
        etascale[RPCb][i] = scales->getRegionalEtaScale(RPCb)->getLowEdge(j);
      }
      etascale[RPCb][33] = scales->getRegionalEtaScale(RPCb)->getScaleMax();
      netascale[RPCb]=33;
    }
    // CSCTF etascale
    {
      etascale[CSCTF][0] = (-1) * scales->getRegionalEtaScale(CSCTF)->getScaleMax();
      int i=1;
      for(int j=31; j>=0; j--,i++) {
        etascale[CSCTF][i] = (-1) * scales->getRegionalEtaScale(CSCTF)->getLowEdge(j);
      }
      for(int j=0; j<=31; j++,i++) {
        etascale[CSCTF][i] = scales->getRegionalEtaScale(CSCTF)->getLowEdge(j);
      }
      etascale[CSCTF][65] = scales->getRegionalEtaScale(CSCTF)->getScaleMax();
      netascale[CSCTF]=65;
    }
    // RPCf etascale
    {
      int i=0;
      for(int j=48; j<=63; j++,i++) {
        etascale[RPCf][i] = scales->getRegionalEtaScale(RPCf)->getLowEdge(j);
      }
      for(int j=0; j<=16; j++,i++) {
        etascale[RPCf][i] = scales->getRegionalEtaScale(RPCf)->getLowEdge(j);
      }
      etascale[RPCf][33] = scales->getRegionalEtaScale(RPCf)->getScaleMax();
      netascale[RPCf]=33;
    }
    // GMT etascale
    {
      etascale[GMT][0] = (-1) * scales->getGMTEtaScale()->getScaleMax();
      int i=1;
      for(int j=30; j>0; j--,i++) {
        etascale[GMT][i] = (-1) * scales->getGMTEtaScale()->getLowEdge(j);
      }
      for(int j=0; j<=30; j++,i++) {
        etascale[GMT][i] = scales->getGMTEtaScale()->getLowEdge(j);
      }
      etascale[GMT][62] = scales->getGMTEtaScale()->getScaleMax();
      netascale[GMT]=62;
    }
    
    
    std::string hname("");
    std::string htitle("");
    
    for(int i=0; i<5; i++) {
      
      hname = subs[i] + "_nbx"; htitle = subs[i] + " multiplicity in bx";
      subs_nbx[i] = dbe->book2D(hname.data(),htitle.data(), 4, 1., 5., 5, -2.5, 2.5);
      subs_nbx[i]->setAxisTitle(subs[i] + " candidates",1);
      subs_nbx[i]->setAxisTitle("bx wrt L1A",2);
      
      hname = subs[i] + "_eta"; htitle = subs[i] + " eta value";
      subs_eta[i] = dbe->book1D(hname.data(),htitle.data(), netascale[i], etascale[i]);
      subs_eta[i]->setAxisTitle("eta",1);
      
      hname = subs[i] + "_phi"; htitle = subs[i] + " phi value";
      subs_phi[i] = dbe->book1D(hname.data(),htitle.data(), nphi, phimin, phimax);
      subs_phi[i]->setAxisTitle("phi (deg)",1);
      
      hname = subs[i] + "_pt"; htitle = subs[i] + " pt value";
      subs_pt[i]  = dbe->book1D(hname.data(),htitle.data(), 31, ptscale);
      subs_pt[i]->setAxisTitle("L1 pT (GeV)",1);
      
      hname = subs[i] + "_qty"; htitle = subs[i] + " qty value";
      subs_qty[i] = dbe->book1D(hname.data(),htitle.data(), nqty, qtymin, qtymax);
      subs_qty[i]->setAxisTitle(subs[i] + " quality",1);
      
      hname = subs[i] + "_etaphi"; htitle = subs[i] + " phi vs eta";
      subs_etaphi[i] = dbe->book2D(hname.data(),htitle.data(), netascale[i], etascale[i], 144, phiscale);
      subs_etaphi[i]->setAxisTitle("eta",1);
      subs_etaphi[i]->setAxisTitle("phi (deg)",2);
      
      hname = subs[i] + "_etaqty"; htitle = subs[i] + " qty vs eta";
      subs_etaqty[i] = dbe->book2D(hname.data(),htitle.data(), netascale[i], etascale[i], 8, qscale);
      subs_etaqty[i]->setAxisTitle("eta",1);
      subs_etaqty[i]->setAxisTitle(subs[i] + " quality",2);
      
      hname = subs[i] + "_bits"; htitle = subs[i] + " bit population";
      subs_bits[i] = dbe->book1D(hname.data(),htitle.data(), 32, -0.5, 31.5);
      subs_bits[i]->setAxisTitle("bit number",1);
      
      hname = subs[i] + "_candlumi"; htitle = "number of " + subs[i] + " candidates per lumisegment";
      subs_candlumi[i] = dbe->book1D(hname.data(),htitle.data(), 250, 0., 250.);
      subs_candlumi[i]->setAxisTitle("luminosity segment number",1);
    }
    
    regional_triggers = dbe->book1D("Regional_trigger","Muon trigger contribution", 4, 0., 4.);
    regional_triggers->setAxisTitle("regional trigger",1);
    regional_triggers->setBinLabel(1,"DTTF",1);
    regional_triggers->setBinLabel(2,"RPCb",1);
    regional_triggers->setBinLabel(3,"CSCTF",1);
    regional_triggers->setBinLabel(4,"RPCf",1);
    
    bx_number = dbe->book1D("Bx_Number","Bx number ROP chip", 3564, 0., 3564.);
    bx_number->setAxisTitle("bx number",1);
    
    dbx_chip = dbe->bookProfile("dbx_Chip","bx count difference wrt ROP chip", 5, 0., 5.,100,-4000.,4000.,"i");
    dbx_chip->setAxisTitle("chip name",1);
    dbx_chip->setAxisTitle("delta bx",2);
    dbx_chip->setBinLabel(1,"IND",1);
    dbx_chip->setBinLabel(2,"INB",1);
    dbx_chip->setBinLabel(3,"INC",1);
    dbx_chip->setBinLabel(4,"INF",1);
    dbx_chip->setBinLabel(5,"SRT",1);
    
    eta_dtcsc_and_rpc = dbe->book1D("eta_DTCSC_and_RPC","eta of confirmed GMT candidates",
        netascale[GMT], etascale[GMT]);
    eta_dtcsc_and_rpc->setAxisTitle("eta",1);
    
    eta_dtcsc_only = dbe->book1D("eta_DTCSC_only","eta of unconfirmed DT/CSC candidates",
        netascale[GMT], etascale[GMT]);
    eta_dtcsc_only->setAxisTitle("eta",1);
    
    eta_rpc_only = dbe->book1D("eta_RPC_only","eta of unconfirmed RPC candidates",
        netascale[GMT], etascale[GMT]);
    eta_rpc_only->setAxisTitle("eta",1);
    
    phi_dtcsc_and_rpc = dbe->book1D("phi_DTCSC_and_RPC","phi of confirmed GMT candidates",
        nphi, phimin, phimax);
    phi_dtcsc_and_rpc->setAxisTitle("phi (deg)",1);
    
    phi_dtcsc_only = dbe->book1D("phi_DTCSC_only","phi of unconfirmed DT/CSC candidates",
        nphi, phimin, phimax);
    phi_dtcsc_only->setAxisTitle("phi (deg)",1);
    
    phi_rpc_only = dbe->book1D("phi_RPC_only","phi of unconfirmed RPC candidates",
        nphi, phimin, phimax);
    phi_rpc_only->setAxisTitle("phi (deg)",1);
    
    etaphi_dtcsc_and_rpc = dbe->book2D("etaphi_DTCSC_and_RPC","eta vs phi map of confirmed GMT candidates",
        100, -2.5, 2.5, nphi, phimin, phimax);
    etaphi_dtcsc_and_rpc->setAxisTitle("eta",1);
    etaphi_dtcsc_and_rpc->setAxisTitle("phi (deg)",2);
    
    etaphi_dtcsc_only = dbe->book2D("etaphi_DTCSC_only","eta vs phi map of unconfirmed DT/CSC candidates",
        100, -2.5, 2.5, nphi, phimin, phimax);
    etaphi_dtcsc_only->setAxisTitle("eta",1);
    etaphi_dtcsc_only->setAxisTitle("phi (deg)",2);
    
    etaphi_rpc_only = dbe->book2D("etaphi_RPC_only","eta vs phi map of unconfirmed RPC candidates",
        100, -2.5, 2.5, nphi, phimin, phimax);
    etaphi_rpc_only->setAxisTitle("eta",1);
    etaphi_rpc_only->setAxisTitle("phi (deg)",2);
    
    
    dist_phi_dt_rpc = dbe->book1D("dist_phi_DT_RPC","Dphi between DT and RPC candidates", 100, -125., 125.);
    dist_phi_dt_rpc->setAxisTitle("delta phi (deg)",1);

    dist_phi_csc_rpc = dbe->book1D("dist_phi_CSC_RPC","Dphi between CSC and RPC candidates", 100, -125., 125.);
    dist_phi_csc_rpc->setAxisTitle("delta phi (deg)",1);

    dist_phi_dt_csc = dbe->book1D("dist_phi_DT_CSC","Dphi between DT and CSC candidates", 100, -125., 125.);
    dist_phi_dt_csc->setAxisTitle("delta phi (deg)",1);
       

    dist_eta_dt_rpc = dbe->book1D("dist_eta_DT_RPC","Deta between DT and RPC candidates", 40, -1., 1.);
    dist_eta_dt_rpc->setAxisTitle("delta eta",1);

    dist_eta_csc_rpc = dbe->book1D("dist_eta_CSC_RPC","Deta between CSC and RPC candidates", 40, -1., 1.);
    dist_eta_csc_rpc->setAxisTitle("delta eta",1);

    dist_eta_dt_csc = dbe->book1D("dist_eta_DT_CSC","Deta between DT and CSC candidates", 40, -1., 1.);
    dist_eta_dt_csc->setAxisTitle("delta eta",1);

       
    n_rpcb_vs_dttf  = dbe->book2D("n_RPCb_vs_DTTF",  "n cands RPCb vs DTTF",  5, -0.5, 4.5, 5, -0.5, 4.5);
    n_rpcb_vs_dttf->setAxisTitle("DTTF candidates",1);
    n_rpcb_vs_dttf->setAxisTitle("barrel RPC candidates",2);
    
    n_rpcf_vs_csctf = dbe->book2D("n_RPCf_vs_CSCTF", "n cands RPCf vs CSCTF", 5, -0.5, 4.5, 5, -0.5, 4.5);
    n_rpcf_vs_csctf->setAxisTitle("CSCTF candidates",1);
    n_rpcf_vs_csctf->setAxisTitle("endcap RPC candidates",2);
    
    n_csctf_vs_dttf = dbe->book2D("n_CSCTF_vs_DTTF", "n cands CSCTF vs DTTF", 5, -0.5, 4.5, 5, -0.5, 4.5);
    n_csctf_vs_dttf->setAxisTitle("DTTF candidates",1);
    n_csctf_vs_dttf->setAxisTitle("CSCTF candidates",2);
    
    
    for(int i=0; i<4; i++) {
      hname = subs[i] + "_dbx"; htitle = "dBx " + subs[i] + " to previous event";
      subs_dbx[i] = dbe->book2D(hname.data(),htitle.data(), 100, 0., 100., 4, 0., 4.);
      for(int j=0; j<4; j++) {
        subs_dbx[i]->setBinLabel((j+1),subs[j].data(),2);
      }
    }        
  }  
}


void L1TGMT::endJob(void)
{
  if(verbose_) cout << "L1TGMT: end job...." << endl;
  LogInfo("EndJob") << "analyzed " << nev_ << " events"; 

 if ( outputFile_.size() != 0  && dbe ) dbe->save(outputFile_);

 return;
}

void L1TGMT::analyze(const Event& e, const EventSetup& c)
{
  nev_++; 
  if(verbose_) cout << "L1TGMT: analyze...." << endl;


  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  e.getByLabel(gmtSource_,pCollection);
  
  if (!pCollection.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1MuGMTReadoutCollection with label "
    << gmtSource_.label() ;
    return;
  }

  // get GMT readout collection
  L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
  // get record vector
  vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  // loop over records of individual bx's
  vector<L1MuGMTReadoutRecord>::const_iterator RRItr;
  for( RRItr = gmt_records.begin(); RRItr != gmt_records.end(); RRItr++ ) 
  {
    
    vector<L1MuRegionalCand> INPCands[4] = {
        RRItr->getDTBXCands(),
        RRItr->getBrlRPCCands(),
        RRItr->getCSCCands(),
        RRItr->getFwdRPCCands()
    };
    vector<L1MuGMTExtendedCand> GMTCands   = RRItr->getGMTCands();
    
    vector<L1MuRegionalCand>::const_iterator INPItr;
    vector<L1MuGMTExtendedCand>::const_iterator GMTItr;
    vector<L1MuGMTExtendedCand>::const_iterator GMTItr2;
    
    int BxInEvent = RRItr->getBxInEvent();
    
    // count non-empty candidates
    int nSUBS[5] = {0, 0, 0, 0, 0};
    for(int i=0; i<4; i++) {
      for( INPItr = INPCands[i].begin(); INPItr != INPCands[i].end(); ++INPItr ) {
        if(!INPItr->empty()) nSUBS[i]++;
      }      
      subs_nbx[i]->Fill(float(nSUBS[i]),float(BxInEvent));
    }

    for( GMTItr = GMTCands.begin(); GMTItr != GMTCands.end(); ++GMTItr ) {
      if(!GMTItr->empty()) nSUBS[GMT]++;
    }
    subs_nbx[GMT]->Fill(float(nSUBS[GMT]),float(BxInEvent));
    
////////////////////////////////////////////////////////////////////////////////////////////
    // from here care only about the L1A bunch crossing
    if(BxInEvent!=0) continue;
    
    // get the absolute bx number of the L1A
    int Bx = RRItr->getBxNr();
    int Ev = RRItr->getEvNr();
    
    bx_number->Fill(double(Bx));
 
    for(int i=0; i<4; i++) {
      for( INPItr = INPCands[i].begin(); INPItr != INPCands[i].end(); ++INPItr ) {
        if(INPItr->empty()) continue;
        subs_eta[i]->Fill(INPItr->etaValue());
        subs_phi[i]->Fill(piconv_*INPItr->phiValue());
        subs_pt[i]->Fill(INPItr->ptValue());
        subs_qty[i]->Fill(INPItr->quality());
        subs_etaphi[i]->Fill(INPItr->etaValue(),piconv_*INPItr->phiValue());
        subs_etaqty[i]->Fill(INPItr->etaValue(),INPItr->quality());
        int word = INPItr->getDataWord();
        for( int j=0; j<32; j++ ) {
          if( word&(1<<j) ) subs_bits[i]->Fill(float(j));
        }
      }
      subs_candlumi[i]->Fill(float(e.luminosityBlock()),float(nSUBS[i]));
    }
        
    for( GMTItr = GMTCands.begin(); GMTItr != GMTCands.end(); ++GMTItr ) {
      if(GMTItr->empty()) continue;
      subs_eta[GMT]->Fill(GMTItr->etaValue());
      subs_phi[GMT]->Fill(piconv_*GMTItr->phiValue());
      subs_pt[GMT]->Fill(GMTItr->ptValue());
      subs_qty[GMT]->Fill(GMTItr->quality());
      subs_etaphi[GMT]->Fill(GMTItr->etaValue(),piconv_*GMTItr->phiValue());
      subs_etaqty[GMT]->Fill(GMTItr->etaValue(),GMTItr->quality());
      int word = GMTItr->getDataWord();
      for( int j=0; j<32; j++ ) {
        if( word&(1<<j) ) subs_bits[GMT]->Fill(float(j));
      }
      
      if(GMTItr->isMatchedCand()) {
        if(GMTItr->quality()>3) {
          eta_dtcsc_and_rpc->Fill(GMTItr->etaValue());
          phi_dtcsc_and_rpc->Fill(piconv_*GMTItr->phiValue());
          etaphi_dtcsc_and_rpc->Fill(GMTItr->etaValue(),piconv_*GMTItr->phiValue());
        }
      } else if(GMTItr->isRPC()) {
        if(GMTItr->quality()>3) {
          eta_rpc_only->Fill(GMTItr->etaValue());
          phi_rpc_only->Fill(piconv_*GMTItr->phiValue());
          etaphi_rpc_only->Fill(GMTItr->etaValue(),piconv_*GMTItr->phiValue());        
        }
      } else {
        if(GMTItr->quality()>3) {
          eta_dtcsc_only->Fill(GMTItr->etaValue());
          phi_dtcsc_only->Fill(piconv_*GMTItr->phiValue());
          etaphi_dtcsc_only->Fill(GMTItr->etaValue(),piconv_*GMTItr->phiValue());
        }
        
        if(GMTItr != GMTCands.end()){
          for( GMTItr2 = GMTCands.begin(); GMTItr2 != GMTCands.end(); ++GMTItr2 ) {
            if(GMTItr2==GMTItr) continue;
            if(GMTItr2->empty()) continue;
            if(GMTItr2->isRPC()) {
              if(GMTItr->isFwd()) {
                dist_eta_csc_rpc->Fill( GMTItr->etaValue() - GMTItr2->etaValue() );
                dist_phi_csc_rpc->Fill( piconv_*(GMTItr->phiValue() - GMTItr2->phiValue()) );
              } else {
                dist_eta_dt_rpc->Fill( GMTItr->etaValue() - GMTItr2->etaValue() );
                dist_phi_dt_rpc->Fill( piconv_*(GMTItr->phiValue() - GMTItr2->phiValue()) );                
              }
            } else {
              if(!(GMTItr->isFwd()) && GMTItr2->isFwd()) {
                dist_eta_dt_csc->Fill( GMTItr->etaValue() - GMTItr2->etaValue() );
                dist_phi_dt_csc->Fill( piconv_*(GMTItr->phiValue() - GMTItr2->phiValue()) );
              } else if(GMTItr->isFwd() && !(GMTItr2->isFwd())){
                dist_eta_dt_csc->Fill( GMTItr2->etaValue() - GMTItr->etaValue() );
                dist_phi_dt_csc->Fill( piconv_*(GMTItr2->phiValue() - GMTItr->phiValue()) );                
              }
            }
          }     
        }
        
      }
      
    }
    subs_candlumi[GMT]->Fill(float(e.luminosityBlock()),float(nSUBS[GMT]));
    
    n_rpcb_vs_dttf ->Fill(float(nSUBS[DTTF]),float(nSUBS[RPCb]));
    n_rpcf_vs_csctf->Fill(float(nSUBS[CSCTF]),float(nSUBS[RPCf]));
    n_csctf_vs_dttf->Fill(float(nSUBS[DTTF]),float(nSUBS[CSCTF]));
    
    regional_triggers->Fill(-1.); // fill underflow for normalization
    for(int i=0; i<4; i++) {
      if(nSUBS[i]) regional_triggers->Fill(float(i));
    }
    
    // fill only if previous event corresponds to previous trigger
    if( (Ev - evnum_old_) == 1 && bxnum_old_ > -1 ) {
      int dBx = Bx - bxnum_old_;
      for(int id = 0; id<4; id++) {
        if( trsrc_old_&(1<<id) ) {
          for(int i=0; i<4; i++) {
            if(nSUBS[i]) subs_dbx[i]->Fill(float(dBx),float(id));
          }
        }
      }
      
    }
    
    // save quantities for the next event
    evnum_old_ = Ev;
    bxnum_old_ = Bx;
    trsrc_old_ = 0;
    for(int i=0; i<4; i++) {
      if(nSUBS[i])  trsrc_old_ |= (1<<i);
    }
    
  }

}


