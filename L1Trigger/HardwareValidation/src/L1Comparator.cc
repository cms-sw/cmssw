#include "L1Trigger/HardwareValidation/interface/L1Comparator.h"
 
L1Comparator::L1Comparator(const edm::ParameterSet& iConfig) {

  ETP_data_Label_ = iConfig.getParameter<edm::InputTag>("ETP_dataLabel");
  ETP_emul_Label_ = iConfig.getParameter<edm::InputTag>("ETP_emulLabel");

  HTP_data_Label_ = iConfig.getParameter<edm::InputTag>("HTP_dataLabel");
  HTP_emul_Label_ = iConfig.getParameter<edm::InputTag>("HTP_emulLabel");

  RCT_data_Label_ = iConfig.getParameter<edm::InputTag>("RCT_dataLabel");
  RCT_emul_Label_ = iConfig.getParameter<edm::InputTag>("RCT_emulLabel");

  GCT_data_Label_ = iConfig.getParameter<edm::InputTag>("GCT_dataLabel");
  GCT_emul_Label_ = iConfig.getParameter<edm::InputTag>("GCT_emulLabel");

  DTP_data_Label_ = iConfig.getParameter<edm::InputTag>("DTP_dataLabel");
  DTP_emul_Label_ = iConfig.getParameter<edm::InputTag>("DTP_emulLabel");
  DTF_data_Label_ = iConfig.getParameter<edm::InputTag>("DTF_dataLabel");
  DTF_emul_Label_ = iConfig.getParameter<edm::InputTag>("DTF_emulLabel");
  	
  CTP_data_Label_ = iConfig.getParameter<edm::InputTag>("CTP_dataLabel");
  CTP_emul_Label_ = iConfig.getParameter<edm::InputTag>("CTP_emulLabel");
  CTF_data_Label_ = iConfig.getParameter<edm::InputTag>("CTF_dataLabel");
  CTF_emul_Label_ = iConfig.getParameter<edm::InputTag>("CTF_emulLabel");

  RPC_data_Label_ = iConfig.getParameter<edm::InputTag>("RPC_dataLabel");
  RPC_emul_Label_ = iConfig.getParameter<edm::InputTag>("RPC_emulLabel");

  LTC_data_Label_ = iConfig.getParameter<edm::InputTag>("LTC_dataLabel");
  LTC_emul_Label_ = iConfig.getParameter<edm::InputTag>("LTC_emulLabel");

  GMT_data_Label_ = iConfig.getParameter<edm::InputTag>("GMT_dataLabel");
  GMT_emul_Label_ = iConfig.getParameter<edm::InputTag>("GMT_emulLabel");

  GT_data_Label_  = iConfig.getParameter<edm::InputTag>("GT_dataLabel");
  GT_emul_Label_  = iConfig.getParameter<edm::InputTag>("GT_emulLabel");
  
  std::vector<unsigned int> compColls 
    = iConfig.getUntrackedParameter<std::vector<unsigned int> >("COMPARE_COLLS");
  doEtp_ = (bool)compColls[ETP];
  doHtp_ = (bool)compColls[HTP];
  doRct_ = (bool)compColls[RCT];
  doGct_ = (bool)compColls[GCT];
  doDtp_ = (bool)compColls[DTP];
  doDtf_ = (bool)compColls[DTF];
  doCtp_ = (bool)compColls[CTP];
  doCtf_ = (bool)compColls[CTF];
  doRpc_ = (bool)compColls[RPC];
  doLtc_ = (bool)compColls[LTC];
  doGmt_ = (bool)compColls[GMTi];
  doGt_  = (bool)compColls[GT];
  
  dumpFileName = iConfig.getUntrackedParameter<std::string>("DumpFile");
  dumpFile.open(dumpFileName.c_str(), std::ios::out);
  if(!dumpFile.good())
    throw cms::Exception("L1ComparatorDumpFileOpenError")
      << " L1Comparator::L1Comparator : "
      << " couldn't open dump file " << dumpFileName.c_str() << std::endl;
  dumpMode = iConfig.getUntrackedParameter<int>("DumpMode");  

  all_match = true;
}


L1Comparator::~L1Comparator(){}

void L1Comparator::beginJob(const edm::EventSetup&) {}

void L1Comparator::endJob() {
  
  dumpFile << "\n\n-------\n"
	   << "Global data|emulator agreement: " 
	   << all_match << std::endl;

  dumpFile.close();
}


void
L1Comparator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  static int nevent = -1;
  nevent++;
  dumpFile << "\nEvent: " << nevent << std::endl;

  ///  Get the data and emulated collections 

  // -- ECAL TP [electromagnetic calorimeter trigger primitives]
  edm::Handle<EcalTrigPrimDigiCollection> ecal_tp_data;
  edm::Handle<EcalTrigPrimDigiCollection> ecal_tp_emul;
  if(doEtp_) {
    iEvent.getByLabel(ETP_data_Label_.label(),"", ecal_tp_data);
    iEvent.getByLabel(ETP_emul_Label_.label(),"", ecal_tp_emul);
  }

  // -- HCAL TP [hadronic calorimeter trigger primitives]
  edm::Handle<HcalTrigPrimDigiCollection> hcal_tp_data;
  edm::Handle<HcalTrigPrimDigiCollection> hcal_tp_emul;
  if(doHtp_) {
    iEvent.getByLabel(HTP_data_Label_, hcal_tp_data);
    iEvent.getByLabel(HTP_emul_Label_, hcal_tp_emul);
  }

  // -- RCT [regional calorimeter trigger]
  edm::Handle<L1CaloEmCollection> rct_em_data;
  edm::Handle<L1CaloEmCollection> rct_em_emul;
  edm::Handle<L1CaloRegionCollection> rct_rgn_data;
  edm::Handle<L1CaloRegionCollection> rct_rgn_emul;
  if(doRct_) {
    iEvent.getByLabel(RCT_data_Label_, rct_em_data);
    iEvent.getByLabel(RCT_emul_Label_, rct_em_emul);
    iEvent.getByLabel(RCT_data_Label_, rct_rgn_data);
    iEvent.getByLabel(RCT_emul_Label_, rct_rgn_emul);
  }

  // -- GCT [global calorimeter trigger]
  edm::Handle<L1GctEmCandCollection>  gct_isolaem_data;
  edm::Handle<L1GctEmCandCollection>  gct_isolaem_emul;
  edm::Handle<L1GctEmCandCollection>  gct_noisoem_data;
  edm::Handle<L1GctEmCandCollection>  gct_noisoem_emul;
  edm::Handle<L1GctJetCandCollection> gct_cenjets_data;
  edm::Handle<L1GctJetCandCollection> gct_cenjets_emul;
  edm::Handle<L1GctJetCandCollection> gct_forjets_data;
  edm::Handle<L1GctJetCandCollection> gct_forjets_emul;
  edm::Handle<L1GctJetCandCollection> gct_taujets_data;
  edm::Handle<L1GctJetCandCollection> gct_taujets_emul;
  if(doGct_) {
    iEvent.getByLabel(GCT_data_Label_.label(),"isoEm",   gct_isolaem_data);
    iEvent.getByLabel(GCT_emul_Label_.label(),"isoEm",   gct_isolaem_emul);
    iEvent.getByLabel(GCT_data_Label_.label(),"nonIsoEm",gct_noisoem_data);
    iEvent.getByLabel(GCT_emul_Label_.label(),"nonIsoEm",gct_noisoem_emul);
    iEvent.getByLabel(GCT_data_Label_.label(),"cenJets", gct_cenjets_data);
    iEvent.getByLabel(GCT_emul_Label_.label(),"cenJets", gct_cenjets_emul);
    iEvent.getByLabel(GCT_data_Label_.label(),"forJets", gct_forjets_data);
    iEvent.getByLabel(GCT_emul_Label_.label(),"forJets", gct_forjets_emul);
    iEvent.getByLabel(GCT_data_Label_.label(),"tauJets", gct_taujets_data);
    iEvent.getByLabel(GCT_emul_Label_.label(),"tauJets", gct_taujets_emul);
  }

  // -- DTP [drift tube] trigger primitive

  edm::Handle<L1MuDTChambPhContainer> dtp_ph_data_;
  edm::Handle<L1MuDTChambPhContainer> dtp_ph_emul_;
  edm::Handle<L1MuDTChambThContainer> dtp_th_data_;
  edm::Handle<L1MuDTChambThContainer> dtp_th_emul_;
  if(doDtp_) {
    iEvent.getByLabel(DTP_data_Label_,dtp_ph_data_);
    iEvent.getByLabel(DTP_data_Label_,dtp_ph_emul_);
    iEvent.getByLabel(DTP_data_Label_,dtp_th_data_);
    iEvent.getByLabel(DTP_data_Label_,dtp_th_emul_);
  }
  L1MuDTChambPhDigiCollection const* dtp_ph_data = dtp_ph_data_->getContainer();
  L1MuDTChambPhDigiCollection const* dtp_ph_emul = dtp_ph_emul_->getContainer();
  L1MuDTChambThDigiCollection const* dtp_th_data = dtp_th_data_->getContainer();
  L1MuDTChambThDigiCollection const* dtp_th_emul = dtp_th_emul_->getContainer();
  //typedef std::vector<L1MuDTChambPhDigi>  Phi_Container;
  //typedef std::vector<L1MuDTChambPhDigi>  L1MuDTChambPhDigiCollection; 
  //edm::Handle<L1MuDTChambPhDigiCollection> dtp_ph_data;
  //L1MuDTChambPhDigiCollection dtp_ph_data = static_cast<L1MuDTChambPhContainer::Phi_Container const*> ( dtp_ph_data_->getContainer());


  // -- DTF [drift tube] track finder
  edm::Handle<L1MuRegionalCandCollection> dtf_data;
  edm::Handle<L1MuRegionalCandCollection> dtf_emul;
  edm::Handle<L1MuDTTrackContainer>       dtf_trk_data_;
  edm::Handle<L1MuDTTrackContainer>       dtf_trk_emul_;
  if(doDtf_) {
    iEvent.getByLabel(DTF_data_Label_.label(),"DT",dtf_data);
    iEvent.getByLabel(DTF_emul_Label_.label(),"DT",dtf_emul);
    iEvent.getByLabel(DTF_data_Label_.label(),"DTTF",dtf_trk_data_);
    iEvent.getByLabel(DTF_emul_Label_.label(),"DTTF",dtf_trk_emul_);
  }
  L1MuRegionalCandCollection const* dtf_trk_data = dtf_trk_data_->getContainer();
  L1MuRegionalCandCollection const* dtf_trk_emul = dtf_trk_emul_->getContainer();


  // -- CSC trigger primitive
  edm::Handle<CSCCorrelatedLCTDigiCollection> ctp_data_;
  edm::Handle<CSCCorrelatedLCTDigiCollection> ctp_emul_;
  if(doCtp_) {
    iEvent.getByLabel(CTP_data_Label_,ctp_data_);
    iEvent.getByLabel(CTP_emul_Label_,ctp_emul_);
  }
  //CSCCorrelatedLCTDigiCollection_ * ctp_data__ = 0;
  //CSCCorrelatedLCTDigiCollection_ * ctp_emul__ = 0;
  //typedef CSCCorrelatedLCTDigiCollection::DigiRangeIterator ctpIt;
  //typedef CSCCorrelatedLCTDigiCollection::const_iterator    ctpIt1;
  //std::cout << "debug L1Comparator ctp 0" << std::endl;
  //for (ctpIt ctpItr = ctp_data_->begin(); ctpItr != ctp_data_->end(); ctpItr++) {
  //  CSCCorrelatedLCTDigiCollection::Range ctpRange = ctp_data_->get((*ctpItr).first);
  //  for (ctpIt1 ctpItr1 = ctpRange.first; ctpItr1 != ctpRange.second; ctpItr1++) {
  //    ctp_data__->push_back(*ctpItr1);
  //  }
  //}
  //std::cout << "debug L1Comparator ctp 1" << std::endl; 
  //for (ctpIt ctpItr = ctp_emul_->begin(); ctpItr != ctp_emul_->end(); ctpItr++) {
  //  CSCCorrelatedLCTDigiCollection::Range ctpRange = ctp_emul_->get((*ctpItr).first);
  //  for (ctpIt1 ctpItr1 = ctpRange.first; ctpItr1 != ctpRange.second; ctpItr1++) {
  //    //ctp_emul__->push_back(*ctpItr1);
  //  }
  //}
  //std::cout << "debug L1Comparator ctp 2" << std::endl; 
  //CSCCorrelatedLCTDigiCollection_ const* ctp_data(ctp_data__);
  //CSCCorrelatedLCTDigiCollection_ const* ctp_emul(ctp_emul__);
  //std::cout << "debug L1Comparator ctp 3" << std::endl; 

  // -- CSC track finder
  edm::Handle<L1CSCTrackCollection> ctf_data;
  edm::Handle<L1CSCTrackCollection> ctf_emul;
  //edm::Handle<L1MuRegionalCandCollection> ctf_data;
  //edm::Handle<L1MuRegionalCandCollection> ctf_emul;
  if(doCtf_) {
    iEvent.getByLabel(CTF_data_Label_,ctf_data);
    iEvent.getByLabel(CTF_emul_Label_,ctf_emul);
  }

  // -- RPC [resistive plate chambers] 
  edm::Handle<L1MuRegionalCandCollection> rpc_cen_data;
  edm::Handle<L1MuRegionalCandCollection> rpc_cen_emul;
  edm::Handle<L1MuRegionalCandCollection> rpc_for_data;
  edm::Handle<L1MuRegionalCandCollection> rpc_for_emul;
  if(doRpc_) {
    iEvent.getByLabel(RPC_data_Label_.label(),"RPCb",rpc_cen_data);
    iEvent.getByLabel(RPC_emul_Label_.label(),"RPCb",rpc_cen_emul);
    iEvent.getByLabel(RPC_data_Label_.label(),"RPCf",rpc_for_data);
    iEvent.getByLabel(RPC_emul_Label_.label(),"RPCf",rpc_for_emul);
  }

  // -- LTC [local trigger controler]
  edm::Handle<LTCDigiCollection> ltc_data;
  edm::Handle<LTCDigiCollection> ltc_emul;
  if(doLtc_) {
    iEvent.getByLabel(LTC_data_Label_,ltc_data);
    iEvent.getByLabel(LTC_emul_Label_,ltc_emul);
  }

  // -- GMT [global muon trigger]
  edm::Handle<L1MuGMTCandCollection> gmt_data;
  edm::Handle<L1MuGMTCandCollection> gmt_emul;
  edm::Handle<L1MuGMTReadoutCollection> gmt_rdt_data_;
  edm::Handle<L1MuGMTReadoutCollection> gmt_rdt_emul_;
  if(doGmt_) {
    iEvent.getByLabel(GMT_data_Label_, gmt_data);
    iEvent.getByLabel(GMT_emul_Label_, gmt_emul);
    iEvent.getByLabel(GMT_data_Label_, gmt_rdt_data_);
    iEvent.getByLabel(GMT_emul_Label_, gmt_rdt_emul_);
  }
  //std::cout << "debug L1Comparator gmt 0" << std::endl; 
  // get record for a given bx
  //L1MuGMTReadoutRecordCollection * gmt_rdt_data__ = 0;
  //L1MuGMTReadoutRecordCollection * gmt_rdt_emul__ = 0; 
  //std::cout << "debug L1Comparator gmt 1" << std::endl; 
  //const int NBX = 10; //?
  //for(int nbx=1; nbx<NBX; nbx++) gmt_rdt_data__->push_back(gmt_rdt_data_->getRecord(nbx));
  //for(int nbx=1; nbx<NBX; nbx++) gmt_rdt_emul__->push_back(gmt_rdt_emul_->getRecord(nbx));
  //L1MuGMTReadoutRecordCollection const* gmt_rdt_data(gmt_rdt_data__);
  //L1MuGMTReadoutRecordCollection const* gmt_rdt_emul(gmt_rdt_emul__);
  //std::cout << "debug L1Comparator gmt 2" << std::endl; 

  // -- GT [global trigger]
  edm::Handle<L1GlobalTriggerReadoutRecord> gt_em_data;
  edm::Handle<L1GlobalTriggerReadoutRecord> gt_em_emul;
  if(doGt_) {
    iEvent.getByLabel(GT_data_Label_, gt_em_data);
    iEvent.getByLabel(GT_emul_Label_, gt_em_emul);
  }
  
  etp_match = true;
  htp_match = true;
  rct_match = true;
  gct_match = true;
  dtp_match = true;
  dtf_match = true;
  ctp_match = true;
  ctf_match = true;
  rpc_match = true;
  ltc_match = true;
  gmt_match = true;
  gt_match  = true;

  char dumptofile[1000];
  char ok[10];

  std::cout << "\n\nL1COMPARE debug Event:" << nevent << "\n\n" << std::endl;

  // >>---- Ecal Trigger Primmitive ---- <<  
  if(doEtp_) {
    /// ETP 
    //std::vector<>
    //EcalTrigPrimDigiCollection const* ecal_tp_emul_ = ecal_tp_emul.product();
    //EcalTrigPrimDigiCollection const* ecal_tp_data_ = ecal_tp_data.product();
    ////static_const<EcalTrigPrimDigiCollection const*>(ecal_tp_emul_));
    //DEcompare<EcalTrigPrimDigiCollection> EtpEmCompare(ecal_tp_data_, ecal_tp_emul_);
    DEcompare<EcalTrigPrimDigiCollection> EtpEmCompare(ecal_tp_data, ecal_tp_emul);
    etp_match &= EtpEmCompare.do_compare(dumpFile,dumpMode);
  }

  // >>---- Hcal Trigger Primmitive ---- <<  
  if(doHtp_) {
    /// HTP 
      DEcompare<HcalTrigPrimDigiCollection> HtpEmCompare(hcal_tp_data, hcal_tp_emul);
      htp_match &= HtpEmCompare.do_compare(dumpFile,dumpMode);
  }

  // >>---- RCT ---- <<  
  if(doRct_) {
    /// RCT em
    DEcompare<L1CaloEmCollection>     RctEmCompare(rct_em_data, rct_em_emul);
    rct_match &= RctEmCompare.do_compare(dumpFile,dumpMode);
    /// RCT regions
    DEcompare<L1CaloRegionCollection> RctRgnCompare(rct_rgn_data, rct_rgn_emul);
    rct_match &= RctRgnCompare.do_compare(dumpFile,dumpMode);
    //debug&alternative computations
    //bool rct_match_1 = compareCollections(rct_em_data, rct_em_emul);  
    //bool rct_match_2 = CompareCollections<L1CaloEmCollection>(rct_em_data, rct_em_emul);
  }
  
  // >>---- GCT ---- <<  
  if(doGct_) {
    if(gct_isolaem_data.isValid() || gct_isolaem_emul.isValid()) {
      DEcompare<L1GctEmCandCollection>  GctIsolaEmCompare(gct_isolaem_data, gct_isolaem_emul);
      gct_match &= GctIsolaEmCompare.do_compare(dumpFile,dumpMode);
    }
    if(gct_noisoem_data.isValid() || gct_noisoem_emul.isValid()) {
      DEcompare<L1GctEmCandCollection>  GctNoIsoEmCompare(gct_noisoem_data, gct_noisoem_emul);
      gct_match &= GctNoIsoEmCompare.do_compare(dumpFile,dumpMode);
    }
    if(gct_cenjets_data.isValid() || gct_cenjets_emul.isValid()) {
      DEcompare<L1GctJetCandCollection> GctCenJetsCompare(gct_cenjets_data, gct_cenjets_emul);
      gct_match &= GctCenJetsCompare.do_compare(dumpFile,dumpMode);
    }
    if(gct_forjets_data.isValid() || gct_forjets_emul.isValid()) {
      DEcompare<L1GctJetCandCollection> GctForJetsCompare(gct_forjets_data, gct_forjets_emul);
      gct_match &= GctForJetsCompare.do_compare(dumpFile,dumpMode);
    }
    if(gct_taujets_data.isValid() || gct_taujets_emul.isValid()) {
      DEcompare<L1GctJetCandCollection> GctTauJetsCompare(gct_taujets_data, gct_taujets_emul);
      gct_match &= GctTauJetsCompare.do_compare(dumpFile,dumpMode);
    }
    //debug&alternative computations
    //bool gct_match_1 = compareCollections(gct_em_data, gct_em_emul);  
    //bool gct_match_2 = CompareCollections<L1GctEmCandCollection>(gct_em_data, gct_em_emul);  
  }
 
  // >>---- DTP ---- <<  
  if(doDtp_) {
    DEcompare<L1MuDTChambPhDigiCollection> DtpPhCompare(dtp_ph_data, dtp_ph_emul);
    dtp_match &= DtpPhCompare.do_compare(dumpFile,dumpMode);
    DEcompare<L1MuDTChambThDigiCollection> DtpThCompare(dtp_th_data, dtp_th_emul);
    dtp_match &= DtpThCompare.do_compare(dumpFile,dumpMode);
  }
  // >>---- DTF ---- <<  
  if(doDtf_) {
    DEcompare<L1MuRegionalCandCollection> DtfCompare(dtf_data, dtf_emul);
    dtf_match &= DtfCompare.do_compare(dumpFile,dumpMode);
    DEcompare<L1MuRegionalCandCollection> DtfTrkCompare(dtf_trk_data, dtf_trk_emul);
    dtf_match &= DtfTrkCompare.do_compare(dumpFile,dumpMode);
  }

//  // >>---- CTP ---- <<  
//  if(doCtp_) {
//    DEcompare<CSCCorrelatedLCTDigiCollection_> CtpCompare(ctp_data, ctp_emul);
//    ctp_match &= CtpCompare.do_compare(dumpFile,dumpMode);
//  }

  // >>---- CTF ---- <<  
  if(doCtf_) {
  //DEcompare<L1MuRegionalCandCollection> CtfCompare(ctf_data, ctf_emul);
    DEcompare<L1CSCTrackCollection> CtfCompare(ctf_data, ctf_emul);
    ctf_match &= CtfCompare.do_compare(dumpFile,dumpMode);
  }

  // >>---- RPC ---- <<  
  if(doRpc_) {
    DEcompare<L1MuRegionalCandCollection> RpcCenCompare(rpc_cen_data, rpc_cen_emul);
    rpc_match &= RpcCenCompare.do_compare(dumpFile,dumpMode);
    DEcompare<L1MuRegionalCandCollection> RpcForCompare(rpc_for_data, rpc_for_emul);
    rpc_match &= RpcForCompare.do_compare(dumpFile,dumpMode);
  }

  // >>---- LTC ---- <<  
  if(doLtc_) {
    DEcompare<LTCDigiCollection> LtcCompare(ltc_data, ltc_emul);
    ltc_match &= LtcCompare.do_compare(dumpFile,dumpMode);
  }

  // >>---- GMT ---- <<  
//  if(doGmt_) {
//    DEcompare<L1MuGMTCandCollection> GmtCompare(gmt_data, gmt_emul);
//    gmt_match &= GmtCompare.do_compare(dumpFile,dumpMode);
//    DEcompare<L1MuGMTReadoutRecordCollection> GmtRdtCompare(gmt_rdt_data, gmt_rdt_emul);
//    gmt_match &= GmtRdtCompare.do_compare(dumpFile,dumpMode);
//  }

  // >>---- GT ---- <<  
  if(doGt_) {
    dumpFile << "\n  GT...\n";
    gt_match &= compareCollections(gt_em_data, gt_em_emul);  
    if(gt_match) sprintf(ok,"successful");
    else         sprintf(ok,"failed");
    sprintf(dumptofile,"  ...GT data and emulator comparison: %s\n", ok); 
    dumpFile<<dumptofile;
    //DecisionWord gt_dec_data = gt_em_data->decisionWord();
    //DecisionWord gt_dec_emul = gt_em_emul->decisionWord();
    //DEcompare<DecisionWord> GtCompare(gt_dec_data, gt_dec_emul);
    //gct_match &= GtCompare.do_compare(dumpFile);
  }
  
  dumpFile << std::flush;


  // >>---- Event match? ---- <<  
  evt_match  = true;
  evt_match &= etp_match;
  evt_match &= rct_match;
  evt_match &= gct_match;
  evt_match &= gt_match;
 
  if(evt_match) sprintf(ok,"GOOD :]");
  else      sprintf(ok,"BAD !!!"); 
 
  sprintf(dumptofile,"\n -> event data and emulator match... %s\n", ok);
  dumpFile<<dumptofile;


  // >>---- Global match? ---- <<  
  all_match &= evt_match;

}


//--//--//--//--//--//--//--//--//--//--//--//--//--//--//--//--

// --- empty candidate? ---
bool isEmpty(const EcalTriggerPrimitiveDigi c) {
  return  ((c[c.sampleOfInterest()].raw())==0);
}

bool isEmpty(const L1CaloEmCand c) {
  return  ((c.raw())==0);
}
bool isEmpty(const L1CaloRegion c) {
  return c.et()==0;
  //tbd modify L1CaloRegion constructor: add accessor to data and detid
}
bool isEmpty(const L1GctEmCand c) {
    return c.empty();
}

// --- print candidate ---
std::string Print(const L1CaloEmCand c, int mode=0) {
  if(mode==1)
    return std::string("L1CaloEmCand");
  std::stringstream ss;
  ss << "0x" << std::setw(4) << std::setfill('0') << std::hex //<< showbase 
     << c.raw() 
     << std::setfill(' ') << std::dec << "  "
     << c << std::endl; 
  return ss.str();
}

std::string Print(const L1CaloRegion c, int mode=0) {
  if(mode==1)
    return std::string("L1CaloRegion");
  std::stringstream ss;
  ss << c << std::endl; 
  return ss.str();
}

std::string Print(const L1GctEmCand c, int mode=0) {
  if(mode==1)
    return std::string("L1GctEmCand");
  std::stringstream ss;
  ss << std::setw(4) << std::setfill('0') << std::hex 
     << "0x" << c.raw() 
     << std::setfill(' ') << std::dec << "  "
     << c << std::endl; 
  return ss.str();
}


// --- alternative computations and cross checks ---- 

bool L1Comparator::compareCollections(edm::Handle<L1CaloEmCollection> data, edm::Handle<L1CaloEmCollection> emul) {
  bool match = true;
  int ndata = data->size();
  int nemul = emul->size();
  if(ndata!=nemul && false) {
    match &= false;
    dumpFile << "\t#cand mismatch"
	     << "\tdata: " << ndata
 	     << "\temul: " << nemul
	     << std::endl;
  }
  std::auto_ptr<L1CaloEmCollection> data_good (new L1CaloEmCollection);
  std::auto_ptr<L1CaloEmCollection> emul_good (new L1CaloEmCollection);
  std::auto_ptr<L1CaloEmCollection> data_bad  (new L1CaloEmCollection);
  std::auto_ptr<L1CaloEmCollection> emul_bad  (new L1CaloEmCollection);
  L1CaloEmCollection::const_iterator itd;
  L1CaloEmCollection::iterator ite; 
  //emul_bad->reserve(emul->size());
  //copy(emul->begin(),emul->end(),emul_bad->begin());
  for(L1CaloEmCollection::const_iterator ite = emul->begin(); ite != emul->end(); ite++) 
    emul_bad->push_back(*ite);
  //loop needed to deal with differring order
  DEutils<L1CaloEmCollection> de_utils;
  for(itd = data->begin(); itd != data->end(); itd++) {
    ite = de_utils.de_find(emul_bad->begin(),emul_bad->end(),*itd);
    //ite = find(emul_bad->begin(),emul_bad->end(),*itd);
    /// found data value?
    if(ite!=emul->end()) { 
      data_good->push_back(*itd);
      emul_good->push_back(*ite);
      ite=emul_bad->erase(ite);
    } else {
      data_bad->push_back(*itd);
      match &= false;
    }
  }
  //debug
  //std::cout << "\totherStats:" 
  //     << " data_bad:"  << data_bad ->size()
  //     << " emul_bad:"  << emul_bad ->size()
  //     << " data_good:" << data_good->size()
  //     << " emul_good:" << emul_good->size()
  //     << std::endl;
  for (int i=0; i<(int)data_bad->size(); i++) {
    dumpCandidate(data_bad->at(i),emul_bad->at(i), dumpFile);
  }
  return match;
}

bool
L1Comparator::compareCollections(edm::Handle<L1GctEmCandCollection> data, edm::Handle<L1GctEmCandCollection> emul) {
   
  bool match = true;
  //count candidates
  int ndata = data -> size();
  int nemul = emul -> size();
  if(ndata!=nemul) {
    match &= false;
    dumpFile << " #cand mismatch (4?)"
	     << "\tdata: " << ndata
	     << "\temul: " << nemul
	     << std::endl;
  }
  
  L1GctEmCandCollection::const_iterator itd = data -> begin();
  L1GctEmCandCollection::const_iterator itm = emul -> begin();
  
  for (int i=0; i<4; i++) {
    match &= dumpCandidate(*itd++,*itm++, dumpFile);
  }  
  return match;
}

bool
L1Comparator::compareCollections(edm::Handle<L1GlobalTriggerReadoutRecord> data, edm::Handle<L1GlobalTriggerReadoutRecord> emul) {
  //if(data!=emul) {
  //  data->printGtDecision(dumpFile);
  //  emul->printGtDecision(dumpFile);
  //  return false;
  //}     
  //return true;
  //data->print(); emul->print();
  DecisionWord dword_data = data->decisionWord();
  DecisionWord dword_emul = emul->decisionWord();
  
  bool match = true;
  match &= (*data==*emul);
  //  match &= (dword_data==dword_emul);
  
  std::vector<bool> bad_bits;
  for(int i=0; i<128; i++) {
    if(dword_data[i]!=dword_emul[i]) {
      bad_bits.push_back(i); 
      match &= false;
    }
  }
  std::vector<bool>::iterator itb;
  std::vector<bool>::size_type idx;
  if(!match) {
    dumpFile << "\t mismatch in bits: ";
    for(itb = bad_bits.begin(); itb != bad_bits.end(); itb++) 
      dumpFile << *itb << " ";
    dumpFile << "\n\tdata: ";   
    for(idx=0; idx<dword_data.size(); idx++) {
      dumpFile <<  dword_data[idx];
      if(idx%4==0) dumpFile << " ";
    } 
    dumpFile << "\n\temul: ";
    for(idx=0; idx<dword_emul.size(); idx++) {
      dumpFile <<  dword_emul[idx];
      if(idx%4==0) dumpFile << " ";
    }
    dumpFile << std::endl; 
  }
  return match;
}

bool L1Comparator::dumpCandidate(L1CaloEmCand& dt, L1CaloEmCand& em, std::ostream& s) {
  if( dt.raw() == em.raw()) 
    return 1;
  s<<dt<<std::endl; 
  s<<em<<std::endl<<std::endl;
  return 0;
}

bool 
L1Comparator::dumpCandidate(const L1GctEmCand& dt, const L1GctEmCand& em, std::ostream& s) {
  char nb[20]; sprintf(nb," ");
  char dumptofile[1000];
  sprintf(dumptofile,"  data: %s0x%x\tname:%s\trank:%d\tieta:%2d\tiphi:%2d\n", 
	  nb,dt.raw(),dt.name().c_str(),dt.rank(),dt.etaIndex(),dt.phiIndex());
  s<<dumptofile;
  sprintf(dumptofile,"  emul: %s0x%x\tname:%s\trank:%d\tieta:%2d\tiphi:%2d\n\n", 
	  nb,em.raw(),em.name().c_str(),em.rank(),em.etaIndex(),em.phiIndex());
  s<<dumptofile;
  return 0;
}

template <class myCol> 
  bool L1Comparator::CompareCollections( edm::Handle<myCol> data, edm::Handle<myCol> emul) {
  bool match = true;
  typedef typename myCol::size_type col_sz;
  typedef typename myCol::iterator col_it;
  col_sz ndata = data->size();
  col_sz nemul = emul->size();
  if(ndata!=nemul) {
    match &= false;
    dumpFile << " #cand mismatch!"
	     << "\tdata: " << ndata
	     << "\temul: " << nemul
	     << std::endl;
  }
  col_it itd = data -> begin();
  col_it itm = emul -> begin();
  for (col_sz i=0; i<ndata; i++) {
    match &= dumpCandidate(*itd++,*itm++, dumpFile);
  }  
  return match; 
}
