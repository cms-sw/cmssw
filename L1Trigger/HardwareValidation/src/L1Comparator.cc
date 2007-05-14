#include "L1Trigger/HardwareValidation/interface/L1Comparator.h"
 
L1Comparator::L1Comparator(const edm::ParameterSet& iConfig) {

  ETP_data_Label_ = iConfig.getUntrackedParameter<std::string>("ETP_dataLabel");
  ETP_emul_Label_ = iConfig.getUntrackedParameter<std::string>("ETP_emulLabel");

  HTP_data_Label_ = iConfig.getUntrackedParameter<std::string>("HTP_dataLabel");
  HTP_emul_Label_ = iConfig.getUntrackedParameter<std::string>("HTP_emulLabel");

  RCT_data_Label_ = iConfig.getUntrackedParameter<std::string>("RCT_dataLabel");
  RCT_emul_Label_ = iConfig.getUntrackedParameter<std::string>("RCT_emulLabel");
  
  GCT_data_Label_ = iConfig.getUntrackedParameter<std::string>("GCT_dataLabel");
  GCT_emul_Label_ = iConfig.getUntrackedParameter<std::string>("GCT_emulLabel");
  /*
  DTP_data_Label_ = iConfig.getUntrackedParameter<std::string>("DTP_dataLabel");
  DTP_emul_Label_ = iConfig.getUntrackedParameter<std::string>("DTP_emulLabel");
  DTF_data_Label_ = iConfig.getUntrackedParameter<std::string>("DTF_dataLabel");
  DTF_emul_Label_ = iConfig.getUntrackedParameter<std::string>("DTF_emulLabel");
  
  CTP_data_Label_ = iConfig.getUntrackedParameter<std::string>("CTP_dataLabel");
  CTP_emul_Label_ = iConfig.getUntrackedParameter<std::string>("CTP_emulLabel");
  CTF_data_Label_ = iConfig.getUntrackedParameter<std::string>("CTF_dataLabel");
  CTF_emul_Label_ = iConfig.getUntrackedParameter<std::string>("CTF_emulLabel");
  
  RTP_data_Label_ = iConfig.getUntrackedParameter<std::string>("RTP_dataLabel");
  RTP_emul_Label_ = iConfig.getUntrackedParameter<std::string>("RTP_emulLabel");
  RTF_data_Label_ = iConfig.getUntrackedParameter<std::string>("RTF_dataLabel");
  RTF_emul_Label_ = iConfig.getUntrackedParameter<std::string>("RTF_emulLabel");
  
  LTC_data_Label_ = iConfig.getUntrackedParameter<std::string>("LTC_dataLabel");
  LTC_emul_Label_ = iConfig.getUntrackedParameter<std::string>("LTC_emulLabel");
  
  GMT_data_Label_ = iConfig.getUntrackedParameter<std::string>("GMT_dataLabel");
  GMT_emul_Label_ = iConfig.getUntrackedParameter<std::string>("GMT_emulLabel");
  */
  GT_data_Label_  = iConfig.getUntrackedParameter<std::string>("GT_dataLabel");
  GT_emul_Label_  = iConfig.getUntrackedParameter<std::string>("GT_emulLabel");
  
  std::vector<unsigned int> compColls 
    = iConfig.getUntrackedParameter<std::vector<unsigned int> >("COMPARE_COLLS");
  doEtp_ = (bool)compColls[ETP];
  doHtp_ = (bool)compColls[HTP];
  doRct_ = (bool)compColls[RCT];
  doGct_ = (bool)compColls[GCT];
  /*
  doDtp_ = (bool)compColls[DTP];
  doDtf_ = (bool)compColls[DTF];
  doCtp_ = (bool)compColls[CTP];
  doCtf_ = (bool)compColls[CTF];
  doRtp_ = (bool)compColls[RTP];
  doRtf_ = (bool)compColls[RTF];
  doLtc_ = (bool)compColls[LTC];
  doGmt_ = (bool)compColls[GMT];
  */
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


  //  e.getByLabel(gctSource_.label(), "Tau", l1eTauJets);

  
  ///  Get the data and emulated collections 

  // -- ECAL TP [electromagnetic calorimeter trigger primitives]
  edm::Handle<EcalTrigPrimDigiCollection> ecal_tp_data;
  edm::Handle<EcalTrigPrimDigiCollection> ecal_tp_emul;
  if(doEtp_) {
    iEvent.getByLabel(ETP_data_Label_, "", ecal_tp_data);
    iEvent.getByLabel(ETP_emul_Label_, "", ecal_tp_emul);
  }

  // -- HCAL TP [hadronic calorimeter trigger primitives]
  edm::Handle<HcalTrigPrimDigiCollection> hcal_tp_data;
  edm::Handle<HcalTrigPrimDigiCollection> hcal_tp_emul;
  if(doHtp_) {
    iEvent.getByLabel(HTP_data_Label_, "", hcal_tp_data);
    iEvent.getByLabel(HTP_emul_Label_, "", hcal_tp_emul);
  }

  // -- RCT [regional calorimeter trigger]
  edm::Handle<L1CaloEmCollection> rct_em_data;
  edm::Handle<L1CaloEmCollection> rct_em_emul;
  edm::Handle<L1CaloRegionCollection> rct_rgn_data;
  edm::Handle<L1CaloRegionCollection> rct_rgn_emul;
  if(doRct_) {
    iEvent.getByLabel(RCT_data_Label_, "", rct_em_data);
    iEvent.getByLabel(RCT_emul_Label_, "", rct_em_emul);
    iEvent.getByLabel(RCT_data_Label_, "", rct_rgn_data);
    iEvent.getByLabel(RCT_emul_Label_, "", rct_rgn_emul);
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
    iEvent.getByLabel(GCT_data_Label_,"isoEm",   gct_isolaem_data);
    iEvent.getByLabel(GCT_emul_Label_,"isoEm",   gct_isolaem_emul);
    iEvent.getByLabel(GCT_data_Label_,"nonIsoEm",gct_noisoem_data);
    iEvent.getByLabel(GCT_emul_Label_,"nonIsoEm",gct_noisoem_emul);
    iEvent.getByLabel(GCT_data_Label_,"cenJets", gct_cenjets_data);
    iEvent.getByLabel(GCT_emul_Label_,"cenJets", gct_cenjets_emul);
    iEvent.getByLabel(GCT_data_Label_,"forJets", gct_forjets_data);
    iEvent.getByLabel(GCT_emul_Label_,"forJets", gct_forjets_emul);
    iEvent.getByLabel(GCT_data_Label_,"tauJets", gct_taujets_data);
    iEvent.getByLabel(GCT_emul_Label_,"tauJets", gct_taujets_emul);
  }


  // -- Tbd get remaining collections here

  /*  
  // -- DTT [drift tube trigger]
  edm::Handle<L1MuDTChambPhContainer> dt_ph_data;
  edm::Handle<L1MuDTChambPhContainer> dt_ph_emul;
  edm::Handle<L1MuDTChambThContainer> dt_th_data;
  edm::Handle<L1MuDTChambThContainer> dt_th_emul;
  if(doDtt_) {
    iEvent.getByLabel(DTT_data_Label_,dt_ph_data);
    iEvent.getByLabel(DTT_emul_Label_,dt_ph_emul);
    iEvent.getByLabel(DTT_data_Label_,dt_th_data);
    iEvent.getByLabel(DTT_emul_Label_,dt_th_emul);
  }

  // -- RPC [resistive plate chambers]
  edm::Handle<std::vector<L1MuRegionalCand> > rpc_data;
  edm::Handle<std::vector<L1MuRegionalCand> > rpc_emul;
  if(doRpc_) {
    e.getByLabel(RPC_data_Label_,rpc_data);
    e.getByLabel(RPC_emul_Label_,rpc_emul);
  }

  // -- LTC [local trigger controler]
  edm::Handle<LTCDigiCollection> ltc_data;
  edm::Handle<LTCDigiCollection> ltc_emul;
  if(doLtc_) {
    e.getByLabel(LTC_data_Label_,ltc_data);
    e.getByLabel(LTC_emul_Label_,ltc_emul);
  }

  // -- GMT [global muon trigger]
  edm::Handle<L1MuGMTReadoutCollection> gmt_data;
  edm::Handle<L1MuGMTReadoutCollection> gmt_emul;
  if(doGmt_) {
    iEvent.getByLabel(GMT_data_Label_, gmt_data);
    iEvent.getByLabel(GMT_emul_Label_, gmt_emul);
  }
  */

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
  /*
  dtp_match = true;
  dtf_match = true;
  ctp_match = true;
  ctf_match = true;
  rtp_match = true;
  rtf_match = true;
  ltc_match = true;
  gmt_match = true;
  */
  gt_match  = true;

  char dumptofile[1000];
  char ok[10];

  std::cout << "\n\nL1COMPARE debug Event:" << nevent << "\n\n" << std::endl;

  // >>---- Ecal Trigger Primmitive ---- <<  
  if(doEtp_) {
    /// ETP 
    //std:: cout << "SIZE data:" << ecal_tp_data.size() << " emul:" << ecal_tp_emul.size() << std::endl;
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
    /// GCT em iso
    DEcompare<L1GctEmCandCollection>  GctIsolaEmCompare(gct_isolaem_data, gct_isolaem_emul);
    //DEcompare<L1GctEmCandCollection>  GctNoIsoEmCompare(gct_noisoem_data, gct_noisoem_emul);
    //DEcompare<L1GctJetCandCollection> GctCenJetsCompare(gct_cenjets_data, gct_cenjets_emul);
    //DEcompare<L1GctJetCandCollection> GctForJetsCompare(gct_forjets_data, gct_forjets_emul);
    //DEcompare<L1GctJetCandCollection> GctTauJetsCompare(gct_taujets_data, gct_taujets_emul);

    gct_match &= GctIsolaEmCompare.do_compare(dumpFile,dumpMode);
    //gct_match &= GctNoIsoEmCompare.do_compare(dumpFile,dumpMode);
    //gct_match &= GctCenJetsCompare.do_compare(dumpFile,dumpMode);
    //gct_match &= GctForJetsCompare.do_compare(dumpFile,dumpMode);
    //gct_match &= GctTauJetsCompare.do_compare(dumpFile,dumpMode);

    //debug&alternative computations
    //bool gct_match_1 = compareCollections(gct_em_data, gct_em_emul);  
    //bool gct_match_2 = CompareCollections<L1GctEmCandCollection>(gct_em_data, gct_em_emul);  
  }
 
  // >>---- Tbd include here remaining collections ---- <<  
 

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
