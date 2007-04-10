#include "L1Trigger/HardwareValidation/interface/L1Comparator.h"

using edm::Handle;
using std::endl;

L1Comparator::L1Comparator(const edm::ParameterSet& iConfig) {
  
  RCT_data_Label_ = iConfig.getUntrackedParameter<string>("RCT_dataLabel");
  RCT_emul_Label_ = iConfig.getUntrackedParameter<string>("RCT_emulLabel");
  
  GCT_data_Label_ = iConfig.getUntrackedParameter<string>("GCT_dataLabel");
  GCT_emul_Label_ = iConfig.getUntrackedParameter<string>("GCT_emulLabel");
  
  GT_data_Label_  = iConfig.getUntrackedParameter<string>("GT_dataLabel");
  GT_emul_Label_  = iConfig.getUntrackedParameter<string>("GT_emulLabel");
  
  std::vector<unsigned int> compColls 
    = iConfig.getUntrackedParameter<std::vector<unsigned int> >("COMPARE_COLLS");
  doRct_ = (bool)compColls[RCT];
  doGct_ = (bool)compColls[GCT];
  doGt_  = (bool)compColls[GT];
  
  dumpFileName = iConfig.getUntrackedParameter<string>("DumpFile");
  dumpFile.open(dumpFileName.c_str(), std::ios::out);
  if(!dumpFile.good())
    throw cms::Exception("L1ComparatorDumpFileOpenError")
      << " L1Comparator::L1Comparator : "
      << " couldn't open dump file " << dumpFileName.c_str() << endl;
  
  all_match = true;
}


L1Comparator::~L1Comparator(){}

void L1Comparator::beginJob(const edm::EventSetup&) {}

void L1Comparator::endJob() {
  
  dumpFile << "\n\n-------\n"
	   << "Global data|emulator agreement: " 
	   << all_match << endl;

  dumpFile.close();
}


void
L1Comparator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  static int nevent = -1;
  nevent++;
  dumpFile << "\nEvent: " << nevent << endl;;
  
  // -- Get the data and emulated RCT em candidates
  Handle<L1CaloEmCollection> rct_em_data;
  Handle<L1CaloEmCollection> rct_em_emul;
  if(doRct_) {
    iEvent.getByLabel(RCT_data_Label_, "", rct_em_data);
    iEvent.getByLabel(RCT_emul_Label_, "", rct_em_emul);
  }

  // -- Get the data and emulated RCT regions
  Handle<L1CaloRegionCollection> rct_rgn_data;
  Handle<L1CaloRegionCollection> rct_rgn_emul;
  if(doRct_) {
    iEvent.getByLabel(RCT_data_Label_, "", rct_rgn_data);
    iEvent.getByLabel(RCT_emul_Label_, "", rct_rgn_emul);
  }

  // -- Get the data and emulated GCT em candidates
  Handle<L1GctEmCandCollection> gct_em_data;
  Handle<L1GctEmCandCollection> gct_em_emul;
  if(doGct_) {
    iEvent.getByLabel(GCT_data_Label_,"isoEm",gct_em_data);
    iEvent.getByLabel(GCT_emul_Label_,"isoEm",gct_em_emul);
  }

  // -- Get the data and emulated GCT jet candidates
  Handle<L1GctJetCandCollection> gct_jet_data;
  Handle<L1GctJetCandCollection> gct_jet_emul;
  if(doGct_) {
    iEvent.getByLabel(GCT_data_Label_,"",gct_jet_data);
    iEvent.getByLabel(GCT_emul_Label_,"",gct_jet_emul);
  }

  // -- Tbd get remaining collections here

  // -- Get the data and emulated GT records
  Handle<L1GlobalTriggerReadoutRecord> gt_em_data;
  Handle<L1GlobalTriggerReadoutRecord> gt_em_emul;
  if(doGt_) {
    iEvent.getByLabel(GT_data_Label_, gt_em_data);
    iEvent.getByLabel(GT_emul_Label_, gt_em_emul);
  }

  
  rct_match = true;
  gct_match = true;
  gt_match  = true;

  char dumptofile[1000];
  char ok[10];


  // >>---- RCT ---- <<  
    if(doRct_) {
      /// RCT em
    DEcompare<L1CaloEmCollection>     RctEmCompare(rct_em_data, rct_em_emul);
    rct_match &= RctEmCompare.do_compare(dumpFile,0);

    /// RCT regions
    DEcompare<L1CaloRegionCollection> RctRgnCompare(rct_rgn_data, rct_rgn_emul);
    rct_match &= RctRgnCompare.do_compare(dumpFile,0);

    //debug&alternative computations
    //bool rct_match_1 = compareCollections(rct_em_data, rct_em_emul);  
    //bool rct_match_2 = CompareCollections<L1CaloEmCollection>(rct_em_data, rct_em_emul);
  }
  

  // >>---- GCT ---- <<  
  if(doGct_) {
    /// GCT em
    DEcompare<L1GctEmCandCollection>  GctEmCompare(gct_em_data, gct_em_emul);
    gct_match &= GctEmCompare.do_compare(dumpFile,0);

    /// GCT jets
    DEcompare<L1GctJetCandCollection> GctJetCompare(gct_jet_data, gct_jet_emul);
    gct_match &= GctJetCompare.do_compare(dumpFile,0);

    //debug&alternative computations
    //bool gct_match_1 = compareCollections(gct_em_data, gct_em_emul);  
    //bool gct_match_2 = CompareCollections<L1GctEmCandCollection>(gct_em_data, gct_em_emul);  
  }
 
  // >>---- Tbd include here remaining collections ---- <<  
 

  // >>---- GT ---- <<  
  if(doGt_) {
    typedef L1GlobalTriggerReadoutRecord::DecisionWord DecisionWord;
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
  

  // >>---- Event match? ---- <<  
  evt_match  = true;
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
bool isEmpty(const L1CaloEmCand c) {
  return  ((c.raw())==0);
}
bool isEmpty(const L1CaloRegion c) {
  return c.et()==0;
  // modify L1CaloRegion constructor: add accessor to data and detid!
}
bool isEmpty(const L1GctEmCand c) {
    return c.empty();
}

// --- print candidate ---
std::string Print(const L1CaloEmCand c, int mode=0) {
  if(mode==1)
    return string("L1CaloEmCand");
  std::stringstream ss;
  ss << "0x" << setw(4) << setfill('0') << hex //<< showbase 
     << c.raw() 
     << setfill(' ') << dec << "  "
     << c << endl; 
  return ss.str();
}

std::string Print(const L1CaloRegion c, int mode=0) {
  if(mode==1)
    return string("L1CaloRegion");
  std::stringstream ss;
  ss << c << endl; 
  return ss.str();
}

std::string Print(const L1GctEmCand c, int mode=0) {
  if(mode==1)
    return string("L1GctEmCand");
  std::stringstream ss;
  ss << setw(4) << setfill('0') << hex 
     << "0x" << c.raw() 
     << setfill(' ') << dec << "  "
     << c << endl; 
  return ss.str();
}


// --- alternative computations and cross checks ---- 

bool L1Comparator::compareCollections(Handle<L1CaloEmCollection> data, Handle<L1CaloEmCollection> emul) {
  bool match = true;
  int ndata = data->size();
  int nemul = emul->size();
  if(ndata!=nemul && false) {
    match &= false;
    dumpFile << "\t#cand mismatch"
	     << "\tdata: " << ndata
 	     << "\temul: " << nemul
	     << endl;
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
  //     << endl;
  for (int i=0; i<(int)data_bad->size(); i++) {
    dumpCandidate(data_bad->at(i),emul_bad->at(i), dumpFile);
  }
  return match;
}

bool
L1Comparator::compareCollections(Handle<L1GctEmCandCollection> data, Handle<L1GctEmCandCollection> emul) {
   
  bool match = true;
  //count candidates
  int ndata = data -> size();
  int nemul = emul -> size();
  if(ndata!=nemul) {
    match &= false;
    dumpFile << " #cand mismatch (4?)"
	     << "\tdata: " << ndata
	     << "\temul: " << nemul
	     << endl;
  }
  
  L1GctEmCandCollection::const_iterator itd = data -> begin();
  L1GctEmCandCollection::const_iterator itm = emul -> begin();
  
  for (int i=0; i<4; i++) {
    match &= dumpCandidate(*itd++,*itm++, dumpFile);
  }  
  return match;
}

bool
L1Comparator::compareCollections(Handle<L1GlobalTriggerReadoutRecord> data, Handle<L1GlobalTriggerReadoutRecord> emul) {
  //if(data!=emul) {
  //  data->printGtDecision(dumpFile);
  //  emul->printGtDecision(dumpFile);
  //  return false;
  //}     
  //return true;
  //data->print(); emul->print();
  L1GlobalTriggerReadoutRecord::DecisionWord dword_data = data->decisionWord();
  L1GlobalTriggerReadoutRecord::DecisionWord dword_emul = emul->decisionWord();
  
  bool match = true;
  match &= (*data==*emul);
  //  match &= (dword_data==dword_emul);
  
  vector<bool> bad_bits;
  for(int i=0; i<128; i++) {
    if(dword_data[i]!=dword_emul[i]) {
      bad_bits.push_back(i); 
      match &= false;
    }
  }
  vector<bool>::iterator itb;
  vector<bool>::size_type idx;
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
    dumpFile << endl; 
  }
  return match;
}

bool L1Comparator::dumpCandidate(L1CaloEmCand& dt, L1CaloEmCand& em, ostream& s) {
  if( dt.raw() == em.raw()) 
    return 1;
  s<<dt<<endl; 
  s<<em<<endl<<endl;
  return 0;
}

bool 
L1Comparator::dumpCandidate(const L1GctEmCand& dt, const L1GctEmCand& em, ostream& s) {
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
  bool L1Comparator::CompareCollections( Handle<myCol> data, Handle<myCol> emul) {
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
	     << endl;
  }
  col_it itd = data -> begin();
  col_it itm = emul -> begin();
  for (col_sz i=0; i<ndata; i++) {
    match &= dumpCandidate(*itd++,*itm++, dumpFile);
  }  
  return match; 
}
