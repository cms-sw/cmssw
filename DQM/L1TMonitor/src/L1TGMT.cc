/*
 * \file L1TGMT.cc
 *
 * $Date: 2008/03/14 20:35:46 $
 * $Revision: 1.14 $
 * \author J. Berryhill, I. Mikulec
 *
 */

#include "DQM/L1TMonitor/interface/L1TGMT.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
using namespace edm;

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

  nev_ = 0;
  evnum_old_ = -1;
  bxnum_old_ = -1;

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
    
    int neta=100; double etamin=-2.5; double etamax=2.5;
    int nphi=144; double phimin=  0.; double phimax=6.2832;
    int nptr=300; double ptrmin=  0.; double ptrmax=150.;
    int nqty=  8; double qtymin=-0.5; double qtymax=7.5;
       
    dttf_nbx = dbe->book2D("DTTF_nbx","DTTF multiplicity in bx", 5, 0., 5., 5, -2.5, 2.5);
    dttf_eta = dbe->book1D("DTTF_eta", "DTTF eta value", neta, etamin, etamax);
    dttf_phi = dbe->book1D("DTTF_phi", "DTTF phi value", nphi, phimin, phimax);
    dttf_ptr = dbe->book1D("DTTF_ptr", "DTTF pt value",  nptr, ptrmin, ptrmax);
    dttf_qty = dbe->book1D("DTTF_qty", "DTTF quality",   nqty, qtymin, qtymax);
    dttf_etaphi = dbe->book2D("DTTF_etaphi","DTTF phi vs eta", neta, etamin, etamax, nphi, phimin, phimax);
    dttf_bits = dbe->book1D("DTTF_bits","DTTF bit population", 32, -0.5, 31.5);
    
    csctf_nbx = dbe->book2D("CSCTF_nbx","CSCTF multiplicity in bx", 5, 0., 5., 5, -2.5, 2.5);
    csctf_eta = dbe->book1D("CSCTF_eta", "CSCTF eta value", neta, etamin, etamax);
    csctf_phi = dbe->book1D("CSCTF_phi", "CSCTF phi value", nphi, phimin, phimax);
    csctf_ptr = dbe->book1D("CSCTF_ptr", "CSCTF pt value",  nptr, ptrmin, ptrmax);
    csctf_qty = dbe->book1D("CSCTF_qty", "CSCTF quality",   nqty, qtymin, qtymax);
    csctf_etaphi = dbe->book2D("CSCTF_etaphi","CSCTF phi vs eta", neta, etamin, etamax, nphi, phimin, phimax);
    csctf_bits = dbe->book1D("CSCTF_bits","CSCTF bit population", 32, -0.5, 31.5);
    
    rpcb_nbx = dbe->book2D("RPCb_nbx","RPCb multiplicity in bx", 5, 0., 5., 5, -2.5, 2.5);
    rpcb_eta = dbe->book1D("RPCb_eta", "RPCb eta value", neta, etamin, etamax);
    rpcb_phi = dbe->book1D("RPCb_phi", "RPCb phi value", nphi, phimin, phimax);
    rpcb_ptr = dbe->book1D("RPCb_ptr", "RPCb pt value",  nptr, ptrmin, ptrmax);
    rpcb_qty = dbe->book1D("RPCb_qty", "RPCb quality",   nqty, qtymin, qtymax);
    rpcb_etaphi = dbe->book2D("RPCb_etaphi","RPCb phi vs eta", neta, etamin, etamax, nphi, phimin, phimax);
    rpcb_bits = dbe->book1D("RPCb_bits","RPCb bit population", 32, -0.5, 31.5);
    
    rpcf_nbx = dbe->book2D("RPCf_nbx","RPCf multiplicity in bx", 5, 0., 5., 5, -2.5, 2.5);
    rpcf_eta = dbe->book1D("RPCf_eta", "RPCf eta value", neta, etamin, etamax);
    rpcf_phi = dbe->book1D("RPCf_phi", "RPCf phi value", nphi, phimin, phimax);
    rpcf_ptr = dbe->book1D("RPCf_ptr", "RPCf pt value",  nptr, ptrmin, ptrmax);
    rpcf_qty = dbe->book1D("RPCf_qty", "RPCf quality",   nqty, qtymin, qtymax);
    rpcf_etaphi = dbe->book2D("RPCf_etaphi","RPCf phi vs eta", neta, etamin, etamax, nphi, phimin, phimax);
    rpcf_bits = dbe->book1D("RPCf_bits","RPCf bit population", 32, -0.5, 31.5);
    
    gmt_nbx = dbe->book2D("GMT_nbx","GMT multiplicity in bx", 5, 0., 5., 5, -2.5, 2.5);
    gmt_eta = dbe->book1D("GMT_eta", "GMT eta value", neta, etamin, etamax);
    gmt_phi = dbe->book1D("GMT_phi", "GMT phi value", nphi, phimin, phimax);
    gmt_ptr = dbe->book1D("GMT_ptr", "GMT pt value",  nptr, ptrmin, ptrmax);
    gmt_qty = dbe->book1D("GMT_qty", "GMT quality",   nqty, qtymin, qtymax);
    gmt_etaphi = dbe->book2D("GMT_etaphi","GMT phi vs eta", neta, etamin, etamax, nphi, phimin, phimax);
    gmt_bits = dbe->book1D("GMT_bits","GMT bit population", 32, -0.5, 31.5);

    n_rpcb_vs_dttf  = dbe->book2D("n_RPCb_vs_DTTF",  "n cands RPCb vs DTTF",  5, -0.5, 4.5, 5, -0.5, 4.5);
    n_rpcf_vs_csctf = dbe->book2D("n_RPCf_vs_CSCTF", "n cands RPCf vs CSCTF", 5, -0.5, 4.5, 5, -0.5, 4.5);
    n_csctf_vs_dttf = dbe->book2D("n_CSCTF_vs_DTTF", "n cands CSCTF vs DTTF", 5, -0.5, 4.5, 5, -0.5, 4.5);
    
    dttf_dbx = dbe->book2D("DTTF_dbx","dBX DTTF to previous event", 100, 0., 100., 4, 0., 4.);
    dttf_dbx->setBinLabel(1,"DTTF",2);
    dttf_dbx->setBinLabel(2,"RPCb",2);
    dttf_dbx->setBinLabel(3,"CSCTF",2);
    dttf_dbx->setBinLabel(4,"RPCf",2);
    
    csctf_dbx = dbe->book2D("CSCTF_dbx","dBX CSCTF to previous event", 100, 0., 100., 4, 0., 4.);
    csctf_dbx->setBinLabel(1,"DTTF",2);
    csctf_dbx->setBinLabel(2,"RPCb",2);
    csctf_dbx->setBinLabel(3,"CSCTF",2);
    csctf_dbx->setBinLabel(4,"RPCf",2);
    
    rpcb_dbx = dbe->book2D("RPCb_dbx","dBX RPCb to previous event", 100, 0., 100., 4, 0., 4.);
    rpcb_dbx->setBinLabel(1,"DTTF",2);
    rpcb_dbx->setBinLabel(2,"RPCb",2);
    rpcb_dbx->setBinLabel(3,"CSCTF",2);
    rpcb_dbx->setBinLabel(4,"RPCf",2);
    
    rpcf_dbx = dbe->book2D("RPCf_dbx","dBX RPCf to previous event", 100, 0., 100., 4, 0., 4.);
    rpcf_dbx->setBinLabel(1,"DTTF",2);
    rpcf_dbx->setBinLabel(2,"RPCb",2);
    rpcf_dbx->setBinLabel(3,"CSCTF",2);
    rpcf_dbx->setBinLabel(4,"RPCf",2);
    
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
    
    vector<L1MuRegionalCand> DTTFCands  = RRItr->getDTBXCands();
    vector<L1MuRegionalCand> CSCTFCands = RRItr->getCSCCands();
    vector<L1MuRegionalCand> RPCbCands  = RRItr->getBrlRPCCands();
    vector<L1MuRegionalCand> RPCfCands  = RRItr->getFwdRPCCands();
    vector<L1MuGMTExtendedCand> GMTCands   = RRItr->getGMTCands();
    
    vector<L1MuRegionalCand>::const_iterator DTTFItr;
    vector<L1MuRegionalCand>::const_iterator CSCTFItr;
    vector<L1MuRegionalCand>::const_iterator RPCbItr;
    vector<L1MuRegionalCand>::const_iterator RPCfItr;
    vector<L1MuGMTExtendedCand>::const_iterator GMTItr;
    
    int BxInEvent = RRItr->getBxInEvent();
    
    // count non-empty candidates
    
    int nDTTF = 0;
    for( DTTFItr = DTTFCands.begin(); DTTFItr != DTTFCands.end(); ++DTTFItr ) {
      if(!DTTFItr->empty()) nDTTF++;
    }
    dttf_nbx->Fill(float(nDTTF),float(BxInEvent));

    int nCSCTF = 0;
    for( CSCTFItr = CSCTFCands.begin(); CSCTFItr != CSCTFCands.end(); ++CSCTFItr ) {
      if(!CSCTFItr->empty()) nCSCTF++;
    }
    csctf_nbx->Fill(float(nCSCTF),float(BxInEvent));
    
    int nRPCb = 0;
    for( RPCbItr = RPCbCands.begin(); RPCbItr != RPCbCands.end(); ++RPCbItr ) {
      if(!RPCbItr->empty()) nRPCb++;
    }
    rpcb_nbx->Fill(float(nRPCb),float(BxInEvent));
    
    int nRPCf = 0;
    for( RPCfItr = RPCfCands.begin(); RPCfItr != RPCfCands.end(); ++RPCfItr ) {
      if(!RPCfItr->empty()) nRPCf++;
    }
    rpcf_nbx->Fill(float(nRPCf),float(BxInEvent));
      
    int nGMT = 0;
    for( GMTItr = GMTCands.begin(); GMTItr != GMTCands.end(); ++GMTItr ) {
      if(!GMTItr->empty()) nGMT++;
    }
    gmt_nbx->Fill(float(nGMT),float(BxInEvent));
    
    // from here care only about the L1A bunch crossing
    if(BxInEvent!=0) continue;
    
    // get the absolute bx number of the L1A
    int Bx = RRItr->getBxNr();
    int Ev = RRItr->getEvNr();
 
    for( DTTFItr = DTTFCands.begin(); DTTFItr != DTTFCands.end(); ++DTTFItr ) {
      if(DTTFItr->empty()) continue;
      dttf_eta->Fill(DTTFItr->etaValue());
      dttf_phi->Fill(DTTFItr->phiValue());
      dttf_ptr->Fill(DTTFItr->ptValue());
      dttf_qty->Fill(DTTFItr->quality());
      dttf_etaphi->Fill(DTTFItr->etaValue(),DTTFItr->phiValue());
      int word = DTTFItr->getDataWord();
      for( int j=0; j<32; j++ ) {
        if( word&(1<<j) ) dttf_bits->Fill(float(j));
      }
    }
    
    for( CSCTFItr = CSCTFCands.begin(); CSCTFItr != CSCTFCands.end(); ++CSCTFItr ) {
      if(CSCTFItr->empty()) continue;
      csctf_eta->Fill(CSCTFItr->etaValue());
      csctf_phi->Fill(CSCTFItr->phiValue());
      csctf_ptr->Fill(CSCTFItr->ptValue());
      csctf_qty->Fill(CSCTFItr->quality());
      csctf_etaphi->Fill(CSCTFItr->etaValue(),CSCTFItr->phiValue());
      int word = CSCTFItr->getDataWord();
      for( int j=0; j<32; j++ ) {
        if( word&(1<<j) ) csctf_bits->Fill(float(j));
      }
    }
    
    for( RPCbItr = RPCbCands.begin(); RPCbItr != RPCbCands.end(); ++RPCbItr ) {
      if(RPCbItr->empty()) continue;
      rpcb_eta->Fill(RPCbItr->etaValue());
      rpcb_phi->Fill(RPCbItr->phiValue());
      rpcb_ptr->Fill(RPCbItr->ptValue());
      rpcb_qty->Fill(RPCbItr->quality());
      rpcb_etaphi->Fill(RPCbItr->etaValue(),RPCbItr->phiValue());
      int word = RPCbItr->getDataWord();
      for( int j=0; j<32; j++ ) {
        if( word&(1<<j) ) rpcb_bits->Fill(float(j));
      }
    }
    
    for( RPCfItr = RPCfCands.begin(); RPCfItr != RPCfCands.end(); ++RPCfItr ) {
      if(RPCfItr->empty()) continue;
      rpcf_eta->Fill(RPCfItr->etaValue());
      rpcf_phi->Fill(RPCfItr->phiValue());
      rpcf_ptr->Fill(RPCfItr->ptValue());
      rpcf_qty->Fill(RPCfItr->quality());
      rpcf_etaphi->Fill(RPCfItr->etaValue(),RPCfItr->phiValue());
      int word = RPCfItr->getDataWord();
      for( int j=0; j<32; j++ ) {
        if( word&(1<<j) ) rpcf_bits->Fill(float(j));
      }
    }
    
    for( GMTItr = GMTCands.begin(); GMTItr != GMTCands.end(); ++GMTItr ) {
      if(GMTItr->empty()) continue;
      gmt_eta->Fill(GMTItr->etaValue());
      gmt_phi->Fill(GMTItr->phiValue());
      gmt_ptr->Fill(GMTItr->ptValue());
      gmt_qty->Fill(GMTItr->quality());
      gmt_etaphi->Fill(GMTItr->etaValue(),GMTItr->phiValue());
      int word = GMTItr->getDataWord();
      for( int j=0; j<32; j++ ) {
        if( word&(1<<j) ) gmt_bits->Fill(float(j));
      }
    }
    
    n_rpcb_vs_dttf->Fill(float(nDTTF),float(nRPCb));
    n_rpcf_vs_csctf->Fill(float(nCSCTF),float(nRPCf));
    n_csctf_vs_dttf->Fill(float(nDTTF),float(nCSCTF));
    
    // fill only if previous event corresponds to previous trigger
    if( (Ev - evnum_old_) == 1 && bxnum_old_ > -1 ) {
      int dBx = Bx - bxnum_old_;
      for(int id = 0; id<4; id++) {
        if( trsrc_old_&(1<<id) ) {
          if(nDTTF)   dttf_dbx->Fill(float(dBx),float(id));
          if(nCSCTF) csctf_dbx->Fill(float(dBx),float(id));
          if(nRPCb)   rpcb_dbx->Fill(float(dBx),float(id));
          if(nRPCf)   rpcf_dbx->Fill(float(dBx),float(id));
        }
      }
      
    }
    
    // save quantities for the next event
    evnum_old_ = Ev;
    bxnum_old_ = Bx;
    trsrc_old_ = 0;
    if(nDTTF)  trsrc_old_ |= 1;
    if(nRPCb)  trsrc_old_ |= 2;
    if(nCSCTF) trsrc_old_ |= 4;
    if(nRPCf)  trsrc_old_ |= 8;
    
    
  }

}

