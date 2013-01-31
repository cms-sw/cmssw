#include "L1Trigger/HardwareValidation/interface/L1Comparator.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace dedefs;

L1Comparator::L1Comparator(const edm::ParameterSet& iConfig) {

  verbose_ = iConfig.getUntrackedParameter<int>("VerboseFlag",0);

  if(verbose())
    std::cout << "\nL1COMPARATOR constructor...\n" << std::flush;

  std::vector<unsigned int> dosys(0,DEnsys); 
  dosys = 
    iConfig.getUntrackedParameter<std::vector<unsigned int> >("COMPARE_COLLS", dosys);
  
  if((int)dosys.size()!=DEnsys)
    edm::LogError("L1Comparator") 
      << "wrong selection of systems to be compared\n"
      << "\t the size of the mask COMPARE_COLLS (" << dosys.size() 
      << ") is not " << DEnsys << std::endl;
  assert((int)dosys.size()==DEnsys);
  
  for(int isys=0; isys<DEnsys; isys++)
    if( dosys[isys]!=0 && dosys[isys]!=1 ) 
      throw cms::Exception("Invalid configuration") 
	<< "L1Comparator: comparison flag for system " << isys 
	<< " is non boolean: " << dosys[isys] << ". Exiting.\n";
    
  for(int i=0; i<DEnsys; i++)
    m_doSys[i] = dosys[i];
  
  if(verbose()) {
    std::cout << "[L1Comparator] do sys? ";
    for(int i=0; i<DEnsys; i++)
      std::cout << m_doSys[i] << " ";
    std::cout << std::endl;

    std::cout << "[L1Comparator] list of systems to process: ";
    for(int i=0; i<DEnsys; i++) 
      if(m_doSys[i]) 
	std::cout << SystLabel[i] << " ";
    std::cout << std::endl;
  }

  ///assertions/temporary
  assert(ETP==0); assert(HTP==1); assert(RCT== 2); assert(GCT== 3);
  assert(DTP==4); assert(DTF==5); assert(CTP== 6); assert(CTF== 7);
  assert(RPC==8); assert(LTC==9); assert(GMT==10); assert(GLT==11);

  if(verbose())
    std::cout << "[L1Comparator] debug print collection labels\n";

  m_DEsource[ETP][0] = iConfig.getParameter<edm::InputTag>("ETPsourceData");
  m_DEsource[ETP][1] = iConfig.getParameter<edm::InputTag>("ETPsourceEmul");
  m_DEsource[HTP][0] = iConfig.getParameter<edm::InputTag>("HTPsourceData");
  m_DEsource[HTP][1] = iConfig.getParameter<edm::InputTag>("HTPsourceEmul");
  m_DEsource[RCT][0] = iConfig.getParameter<edm::InputTag>("RCTsourceData");
  m_DEsource[RCT][1] = iConfig.getParameter<edm::InputTag>("RCTsourceEmul");
  m_DEsource[GCT][0] = iConfig.getParameter<edm::InputTag>("GCTsourceData");
  m_DEsource[GCT][1] = iConfig.getParameter<edm::InputTag>("GCTsourceEmul");
  m_DEsource[DTP][0] = iConfig.getParameter<edm::InputTag>("DTPsourceData");
  m_DEsource[DTP][1] = iConfig.getParameter<edm::InputTag>("DTPsourceEmul");
  m_DEsource[DTF][0] = iConfig.getParameter<edm::InputTag>("DTFsourceData");
  m_DEsource[DTF][1] = iConfig.getParameter<edm::InputTag>("DTFsourceEmul");
  m_DEsource[CTP][0] = iConfig.getParameter<edm::InputTag>("CTPsourceData");
  m_DEsource[CTP][1] = iConfig.getParameter<edm::InputTag>("CTPsourceEmul");
  m_DEsource[CTF][0] = iConfig.getParameter<edm::InputTag>("CTFsourceData");
  m_DEsource[CTF][1] = iConfig.getParameter<edm::InputTag>("CTFsourceEmul");
  m_DEsource[CTF][2] = iConfig.getParameter<edm::InputTag>("CTTsourceData");
  m_DEsource[CTF][3] = iConfig.getParameter<edm::InputTag>("CTTsourceEmul");
  m_DEsource[RPC][0] = iConfig.getParameter<edm::InputTag>("RPCsourceData");
  m_DEsource[RPC][1] = iConfig.getParameter<edm::InputTag>("RPCsourceEmul");
  m_DEsource[LTC][0] = iConfig.getParameter<edm::InputTag>("LTCsourceData");
  m_DEsource[LTC][1] = iConfig.getParameter<edm::InputTag>("LTCsourceEmul");
  m_DEsource[GMT][0] = iConfig.getParameter<edm::InputTag>("GMTsourceData");
  m_DEsource[GMT][1] = iConfig.getParameter<edm::InputTag>("GMTsourceEmul");
  m_DEsource[GLT][0] = iConfig.getParameter<edm::InputTag>("GLTsourceData");
  m_DEsource[GLT][1] = iConfig.getParameter<edm::InputTag>("GLTsourceEmul");

  for(int sys=0; sys<DEnsys; sys++) {
    std::string data_label = SystLabel[sys] + "sourceData";
    std::string emul_label = SystLabel[sys] + "sourceEmul";
    //m_DEsource[sys][0] = iConfig.getParameter<edm::InputTag>(data_label);
    //m_DEsource[sys][1] = iConfig.getParameter<edm::InputTag>(emul_label);
    //if(sys==CTF) {
    //  std::string data_label(""); data_label+="CTTsourceData";
    //  std::string emul_label(""); emul_label+="CTTsourceEmul";
    //  m_DEsource[sys][2] = iConfig.getParameter<edm::InputTag>(data_label);
    //  m_DEsource[sys][3] = iConfig.getParameter<edm::InputTag>(emul_label);
    //}
    if(m_doSys[sys] && verbose()) {
      std::cout << " sys:"   << sys << " label:" << SystLabel[sys]  
		<< "\n\tdt:" << data_label << " : " <<m_DEsource[sys][0]
		<< "\n\tem:" << emul_label << " : " <<m_DEsource[sys][1]
		<< std::endl;
      if(sys==CTF) {
	std::cout << "\tdt:"     << data_label << " : " <<m_DEsource[sys][2]
     		  << "\n\tem:" << emul_label << " : " <<m_DEsource[sys][3]
		  << std::endl;
      }
    }
  }

  
  m_fedId = iConfig.getUntrackedParameter<int>("FEDid", 0);
  m_FEDsource[0] = 
    iConfig.getUntrackedParameter<edm::InputTag>("FEDsourceData",edm::InputTag());
  m_FEDsource[1] = 
    iConfig.getUntrackedParameter<edm::InputTag>("FEDsourceEmul",edm::InputTag());


  /// dump level:  -1(all),0(none),1(disagree),2(loc.disagree),3(loc.agree)
  m_dumpMode = iConfig.getUntrackedParameter<int>("DumpMode",0);  
  m_dumpFileName = iConfig.getUntrackedParameter<std::string>("DumpFile","");
  if(m_dumpMode) {
    m_dumpFile.open(m_dumpFileName.c_str(), std::ios::out);
    if(!m_dumpFile.good())
      edm::LogInfo("L1ComparatorDumpFileOpenError")
	<< " L1Comparator::L1Comparator() : "
	<< " couldn't open dump file " << m_dumpFileName.c_str() << std::endl;
  }

  m_match = true;
  dumpEvent_ = true;
  nevt_=-1;

  for(int i=0; i<DEnsys; i++) {
    for(int j=0; j<2; j++) 
      DEncand[i][j] = 0;
    DEmatchEvt[i] = true;
  }

  m_dedigis.clear();
  /// create d|e record product
  produces<L1DataEmulRecord>().setBranchAlias("L1DataEmulRecord");  

  if(verbose())
    std::cout << "\nL1Comparator constructor...done.\n" << std::flush;
}


L1Comparator::~L1Comparator(){}

void L1Comparator::beginJob(void) {}

void L1Comparator::beginRun(edm::Run& iRun, const edm::EventSetup& iSetup) {

  if(verbose())
    std::cout << "\nL1COMPARATOR beginRun...\n" << std::flush;


  // disable subsystem if not included in current run configuration
  try 
    {
      edm::ESHandle< L1TriggerKey > pKey ;
      iSetup.get< L1TriggerKeyRcd >().get( pKey ) ;

      m_doSys[RCT] &= (!(pKey->subsystemKey( L1TriggerKey::kRCT)  .empty()));
      m_doSys[GCT] &= (!(pKey->subsystemKey( L1TriggerKey::kGCT)  .empty()));
      m_doSys[DTF] &= (!(pKey->subsystemKey( L1TriggerKey::kDTTF) .empty()));
      m_doSys[CTF] &= (!(pKey->subsystemKey( L1TriggerKey::kCSCTF).empty()));
      m_doSys[RPC] &= (!(pKey->subsystemKey( L1TriggerKey::kRPC)  .empty()));
      m_doSys[GMT] &= (!(pKey->subsystemKey( L1TriggerKey::kGMT)  .empty()));
      m_doSys[GLT] &= (!(pKey->subsystemKey( L1TriggerKey::kGT)   .empty()));

     if(verbose()) {
	if ( pKey->subsystemKey( L1TriggerKey::kRCT  ).empty() )
	  std::cout << "RCT   key is empty. Sub-systems is disabled ("<<m_doSys[RCT]<<")\n";
	if ( pKey->subsystemKey( L1TriggerKey::kGCT  ).empty() )
	  std::cout << "GCT   key is empty. Sub-systems is disabled ("<<m_doSys[GCT]<<")\n";
	if ( pKey->subsystemKey( L1TriggerKey::kDTTF ).empty() )
	  std::cout << "DTTF  key is empty. Sub-systems is disabled ("<<m_doSys[DTF]<<")\n";
	if ( pKey->subsystemKey( L1TriggerKey::kCSCTF).empty() )
	  std::cout << "CSCTF key is empty. Sub-systems is disabled ("<<m_doSys[CTF]<<")\n";
	if ( pKey->subsystemKey( L1TriggerKey::kRPC  ).empty() )
	  std::cout << "RPC   key is empty. Sub-systems is disabled ("<<m_doSys[RPC]<<")\n";
	if ( pKey->subsystemKey( L1TriggerKey::kGMT  ).empty() )
	  std::cout << "GMT   key is empty. Sub-systems is disabled ("<<m_doSys[GMT]<<")\n";
	if ( pKey->subsystemKey( L1TriggerKey::kGT   ).empty() )
	  std::cout << "GT    key is empty. Sub-systems is disabled ("<<m_doSys[GLT]<<")\n";
	std::cout << "TSC key = " << pKey->tscKey() << std::endl; 
      }

      //access subsystem key if needed, eg:
      //std::cout << "RCT key:" << pKey->subsystemKey( L1TriggerKey::kRCT ) << std::endl;
    } 
  catch( cms::Exception& ex ) 
    {
      edm::LogWarning("L1Comparator") 
	<< "No L1TriggerKey found." 
	<< std::endl;
    }  

  if(verbose())
    std::cout << "L1COMPARATOR beginRun... done\n" << std::flush;

}

void L1Comparator::endJob() {
  if(m_dumpMode)
    m_dumpFile << "\n\n-------\n"
	       << "Global data|emulator agreement: " 
	       << m_match << std::endl;
  m_dumpFile.close();
}

void
L1Comparator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  nevt_++;
  evtNum_ = iEvent.id().event();
  runNum_ = iEvent.id().run();

  if(verbose())
    std::cout << "\nL1COMPARATOR entry:" << nevt_ << " | evt:" << evtNum_ 
	      << " | run:" << runNum_ << "\n" << std::flush;

  //flag whether event id has already been written to dumpFile
  dumpEvent_ = true;

  //reset event holder quantities
  for(int i=0; i<DEnsys; i++) {
    for(int j=0; j<2; j++) 
      DEncand[i][j] = 0;
    DEmatchEvt[i] = true;
  }
  m_dedigis.clear();


  /// --  Get the data and emulated collections -----------------------------

  // -- ETP [electromagnetic calorimeter trigger primitives]
  edm::Handle<EcalTrigPrimDigiCollection> ecal_tp_data;
  edm::Handle<EcalTrigPrimDigiCollection> ecal_tp_emul;
  if(m_doSys[ETP]) {
    iEvent.getByLabel(m_DEsource[ETP][0], ecal_tp_data);
    iEvent.getByLabel(m_DEsource[ETP][1], ecal_tp_emul);
  }

  // -- HTP [hadronic calorimeter trigger primitives]
  edm::Handle<HcalTrigPrimDigiCollection> hcal_tp_data;
  edm::Handle<HcalTrigPrimDigiCollection> hcal_tp_emul;
  if(m_doSys[HTP]) {
    iEvent.getByLabel(m_DEsource[HTP][0], hcal_tp_data);
    iEvent.getByLabel(m_DEsource[HTP][1], hcal_tp_emul);
  }

  // -- RCT [regional calorimeter trigger]
  edm::Handle<L1CaloEmCollection> rct_em_data;
  edm::Handle<L1CaloEmCollection> rct_em_emul;
  edm::Handle<L1CaloRegionCollection> rct_rgn_data;
  edm::Handle<L1CaloRegionCollection> rct_rgn_emul;
  if(m_doSys[RCT]) {
    iEvent.getByLabel(m_DEsource[RCT][0], rct_em_data);
    iEvent.getByLabel(m_DEsource[RCT][1], rct_em_emul);
    iEvent.getByLabel(m_DEsource[RCT][0], rct_rgn_data);
    iEvent.getByLabel(m_DEsource[RCT][1], rct_rgn_emul);
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

  edm::Handle<L1GctEtHadCollection>	  gct_ht_data;
  edm::Handle<L1GctEtHadCollection>	  gct_ht_emul;
  edm::Handle<L1GctEtMissCollection>	  gct_etmiss_data;
  edm::Handle<L1GctEtMissCollection>	  gct_etmiss_emul;
  edm::Handle<L1GctEtTotalCollection>	  gct_ettota_data;
  edm::Handle<L1GctEtTotalCollection>	  gct_ettota_emul;
  edm::Handle<L1GctHtMissCollection>	  gct_htmiss_data;
  edm::Handle<L1GctHtMissCollection>	  gct_htmiss_emul;
  edm::Handle<L1GctHFRingEtSumsCollection>gct_hfring_data;
  edm::Handle<L1GctHFRingEtSumsCollection>gct_hfring_emul;
  edm::Handle<L1GctHFBitCountsCollection> gct_hfbcnt_data;
  edm::Handle<L1GctHFBitCountsCollection> gct_hfbcnt_emul;
  edm::Handle<L1GctJetCountsCollection>	  gct_jetcnt_data;  
  edm::Handle<L1GctJetCountsCollection>	  gct_jetcnt_emul;

  if(m_doSys[GCT]) {
   iEvent.getByLabel(m_DEsource[GCT][0].label(),"isoEm",   gct_isolaem_data);
   iEvent.getByLabel(m_DEsource[GCT][1].label(),"isoEm",   gct_isolaem_emul);
   iEvent.getByLabel(m_DEsource[GCT][0].label(),"nonIsoEm",gct_noisoem_data);
   iEvent.getByLabel(m_DEsource[GCT][1].label(),"nonIsoEm",gct_noisoem_emul);
   iEvent.getByLabel(m_DEsource[GCT][0].label(),"cenJets", gct_cenjets_data);
   iEvent.getByLabel(m_DEsource[GCT][1].label(),"cenJets", gct_cenjets_emul);
   iEvent.getByLabel(m_DEsource[GCT][0].label(),"forJets", gct_forjets_data);
   iEvent.getByLabel(m_DEsource[GCT][1].label(),"forJets", gct_forjets_emul);
   iEvent.getByLabel(m_DEsource[GCT][0].label(),"tauJets", gct_taujets_data);
   iEvent.getByLabel(m_DEsource[GCT][1].label(),"tauJets", gct_taujets_emul);

   iEvent.getByLabel(m_DEsource[GCT][0],gct_ht_data);	  
   iEvent.getByLabel(m_DEsource[GCT][1],gct_ht_emul);	 
   iEvent.getByLabel(m_DEsource[GCT][0],gct_etmiss_data);  
   iEvent.getByLabel(m_DEsource[GCT][1],gct_etmiss_emul);
   iEvent.getByLabel(m_DEsource[GCT][0],gct_ettota_data);	  
   iEvent.getByLabel(m_DEsource[GCT][1],gct_ettota_emul); 
   iEvent.getByLabel(m_DEsource[GCT][0],gct_htmiss_data);  
   iEvent.getByLabel(m_DEsource[GCT][1],gct_htmiss_emul);
   iEvent.getByLabel(m_DEsource[GCT][0],gct_hfring_data);
   iEvent.getByLabel(m_DEsource[GCT][1],gct_hfring_emul);
   iEvent.getByLabel(m_DEsource[GCT][0],gct_hfbcnt_data);
   iEvent.getByLabel(m_DEsource[GCT][1],gct_hfbcnt_emul);
   iEvent.getByLabel(m_DEsource[GCT][0],gct_jetcnt_data);  
   iEvent.getByLabel(m_DEsource[GCT][1],gct_jetcnt_emul);
  }

  // -- DTP [drift tube trigger primitive]
  edm::Handle<L1MuDTChambPhContainer> dtp_ph_data_;
  edm::Handle<L1MuDTChambPhContainer> dtp_ph_emul_;
  edm::Handle<L1MuDTChambThContainer> dtp_th_data_;
  edm::Handle<L1MuDTChambThContainer> dtp_th_emul_;
  if(m_doSys[DTP]) {
    iEvent.getByLabel(m_DEsource[DTP][0],dtp_ph_data_);
    iEvent.getByLabel(m_DEsource[DTP][1],dtp_ph_emul_);
    iEvent.getByLabel(m_DEsource[DTP][0],dtp_th_data_);
    iEvent.getByLabel(m_DEsource[DTP][1],dtp_th_emul_);
  }
  L1MuDTChambPhDigiCollection const* dtp_ph_data = 0; 
  L1MuDTChambPhDigiCollection const* dtp_ph_emul = 0; 
  L1MuDTChambThDigiCollection const* dtp_th_data = 0; 
  L1MuDTChambThDigiCollection const* dtp_th_emul = 0; 

  if(dtp_ph_data_.isValid()) dtp_ph_data = dtp_ph_data_->getContainer();
  if(dtp_ph_emul_.isValid()) dtp_ph_emul = dtp_ph_emul_->getContainer();
  if(dtp_th_data_.isValid()) dtp_th_data = dtp_th_data_->getContainer();
  if(dtp_th_emul_.isValid()) dtp_th_emul = dtp_th_emul_->getContainer();

  // -- DTF [drift tube track finder]
  edm::Handle<L1MuRegionalCandCollection> dtf_data;
  edm::Handle<L1MuRegionalCandCollection> dtf_emul;
  edm::Handle<L1MuDTTrackContainer>       dtf_trk_data_;
  edm::Handle<L1MuDTTrackContainer>       dtf_trk_emul_;
  L1MuRegionalCandCollection const* dtf_trk_data = 0;
  L1MuRegionalCandCollection const* dtf_trk_emul = 0;
  if(m_doSys[DTF]) {
    iEvent.getByLabel(m_DEsource[DTF][0].label(),"DT",dtf_data);
    iEvent.getByLabel(m_DEsource[DTF][1].label(),"DT",dtf_emul);
  //iEvent.getByLabel(m_DEsource[DTF][0].label(),"DTTF",dtf_trk_data_);
    iEvent.getByLabel(m_DEsource[DTF][0].label(),"DATA",dtf_trk_data_);
    iEvent.getByLabel(m_DEsource[DTF][1].label(),"DTTF",dtf_trk_emul_);
  } 
  //extract the regional cands
  typedef std::vector<L1MuDTTrackCand> L1MuDTTrackCandCollection;
  L1MuRegionalCandCollection dtf_trk_data_v, dtf_trk_emul_v;
  dtf_trk_data_v.clear(); dtf_trk_emul_v.clear();
  if(dtf_trk_data_.isValid()) {
    L1MuDTTrackCandCollection *dttc = dtf_trk_data_->getContainer();
    for(L1MuDTTrackCandCollection::const_iterator  it=dttc->begin(); 
	it!=dttc->end(); it++)
      dtf_trk_data_v.push_back(L1MuRegionalCand(*it)); 
  }
  if(dtf_trk_emul_.isValid()) {
    L1MuDTTrackCandCollection *dttc = dtf_trk_emul_->getContainer();
    for(L1MuDTTrackCandCollection::const_iterator  it=dttc->begin(); 
	it!=dttc->end(); it++)
      dtf_trk_emul_v.push_back(L1MuRegionalCand(*it)); 
  }  
  dtf_trk_data =&dtf_trk_data_v;
  dtf_trk_emul =&dtf_trk_emul_v;
  
  // -- CTP [cathode strip chamber trigger primitive]
  edm::Handle<CSCALCTDigiCollection>          ctp_ano_data_;
  edm::Handle<CSCALCTDigiCollection>          ctp_ano_emul_;
  edm::Handle<CSCCLCTDigiCollection>          ctp_cat_data_;
  edm::Handle<CSCCLCTDigiCollection>          ctp_cat_emul_;
  edm::Handle<CSCCorrelatedLCTDigiCollection> ctp_lct_data_;
  edm::Handle<CSCCorrelatedLCTDigiCollection> ctp_lct_emul_;
  CSCALCTDigiCollection_          const* ctp_ano_data = 0;
  CSCALCTDigiCollection_          const* ctp_ano_emul = 0;
  CSCCLCTDigiCollection_          const* ctp_cat_data = 0;
  CSCCLCTDigiCollection_          const* ctp_cat_emul = 0;
  CSCCorrelatedLCTDigiCollection_ const* ctp_lct_data = 0;
  CSCCorrelatedLCTDigiCollection_ const* ctp_lct_emul = 0;
  if(m_doSys[CTP]) {
    if(m_DEsource[CTP][0].label().find("tf")!=std::string::npos) {
      //if correlated LCTs from TF, read needed info from TP data digis
      iEvent.getByLabel("muonCSCDigis", "MuonCSCALCTDigi"     ,ctp_ano_data_);
      iEvent.getByLabel("muonCSCDigis", "MuonCSCCLCTDigi"     ,ctp_cat_data_);
      iEvent.getByLabel(m_DEsource[CTP][0]                    ,ctp_lct_data_);
    } else {
      iEvent.getByLabel(m_DEsource[CTP][0].label(),"MuonCSCALCTDigi",ctp_ano_data_);
      iEvent.getByLabel(m_DEsource[CTP][0].label(),"MuonCSCCLCTDigi",ctp_cat_data_);
      iEvent.getByLabel(m_DEsource[CTP][0].label(),"MuonCSCCorrelatedLCTDigi",ctp_lct_data_);
    }
    iEvent.getByLabel(m_DEsource[CTP][1]                    ,ctp_ano_emul_);
    iEvent.getByLabel(m_DEsource[CTP][1]                    ,ctp_cat_emul_);
    iEvent.getByLabel(m_DEsource[CTP][1]                    ,ctp_lct_emul_);
  }

  ///place candidates into vectors
  //Anode LCT
  CSCALCTDigiCollection_ ctp_ano_data_v, ctp_ano_emul_v;
  ctp_ano_data_v.clear(); ctp_ano_emul_v.clear();
  if(ctp_ano_data_.isValid() && ctp_ano_emul_.isValid()) {
    // The following numbers should come from config. database eventually...
    int fifo_pretrig     = 10;
    int fpga_latency     =  6;
    int l1a_window_width =  7;
    // Time offset of raw hits w.r.t. the full 12-bit BXN.
    int rawhit_tbin_offset =
      (fifo_pretrig - fpga_latency) + (l1a_window_width-1)/2;
    // Extra difference due to additional register stages; determined
    // empirically.
    int register_delay =  2;
    typedef CSCALCTDigiCollection::DigiRangeIterator mapIt;
    typedef CSCALCTDigiCollection::const_iterator    vecIt;
    for (mapIt mit = ctp_ano_data_->begin(); mit != ctp_ano_data_->end(); mit++)
      for (vecIt vit = ctp_ano_data_->get((*mit).first).first; 
	   vit != ctp_ano_data_->get((*mit).first).second; vit++) 
	ctp_ano_data_v.push_back(*vit);
    for (mapIt mit = ctp_ano_emul_->begin(); mit != ctp_ano_emul_->end(); mit++)
      for (vecIt vit = ctp_ano_emul_->get((*mit).first).first; 
	   vit != ctp_ano_emul_->get((*mit).first).second; vit++) {
	int emul_bx_corr =
	  (*vit).getBX() - rawhit_tbin_offset + register_delay;
	CSCALCTDigi alct((*vit).isValid(),        (*vit).getQuality(),
			 (*vit).getAccelerator(), (*vit).getCollisionB(),
			 (*vit).getKeyWG(),       emul_bx_corr,
			 (*vit).getTrknmb()); 
	ctp_ano_emul_v.push_back(alct);
      }
  }
  ctp_ano_data =&ctp_ano_data_v;
  ctp_ano_emul =&ctp_ano_emul_v;
  //Cathode LCT
  CSCCLCTDigiCollection_ ctp_cat_data_v, ctp_cat_emul_v;
  ctp_cat_data_v.clear(); ctp_cat_emul_v.clear();
  if(ctp_cat_data_.isValid() && ctp_cat_emul_.isValid()) {
    int tbin_cathode_offset = 7, emul_bx_corr;
    typedef CSCCLCTDigiCollection::DigiRangeIterator mapIt;
    typedef CSCCLCTDigiCollection::const_iterator    vecIt;
    for (mapIt mit = ctp_cat_data_->begin(); mit != ctp_cat_data_->end(); mit++)
      for (vecIt vit = ctp_cat_data_->get((*mit).first).first; 
	   vit != ctp_cat_data_->get((*mit).first).second; vit++) 
	ctp_cat_data_v.push_back(*vit);
    for (mapIt mit = ctp_cat_emul_->begin(); mit != ctp_cat_emul_->end(); mit++) {
      const CSCDetId& detid = (*mit).first;

      // Extract full 12-bit BX word from CLCT data collections.
      int full_cathode_bx = -999;
      const CSCCLCTDigiCollection::Range& crange = ctp_cat_data_->get(detid);
      for (vecIt digiIt = crange.first; digiIt != crange.second; digiIt++) {
	if ((*digiIt).isValid()) {
	  full_cathode_bx = (*digiIt).getFullBX();
	  break;
	}
      }

      for (vecIt vit = ctp_cat_emul_->get(detid).first; 
	   vit != ctp_cat_emul_->get(detid).second; vit++) {
	int emul_bx = (*vit).getBX();
	if (full_cathode_bx != -999)
	  emul_bx_corr =
	    (full_cathode_bx + emul_bx - tbin_cathode_offset) & 0x03;
	else
	  emul_bx_corr = emul_bx & 0x03;
	CSCCLCTDigi clct((*vit).isValid(),    (*vit).getQuality(),
			 (*vit).getPattern(), (*vit).getStripType(),
			 (*vit).getBend(),    (*vit).getStrip(),
			 (*vit).getCFEB(),    emul_bx_corr,
			 (*vit).getTrknmb());
	ctp_cat_emul_v.push_back(clct);
      }
    }
  }
  ctp_cat_data =&ctp_cat_data_v;
  ctp_cat_emul =&ctp_cat_emul_v;
  //Correlated (anode+cathode) LCTs
  CSCCorrelatedLCTDigiCollection_ ctp_lct_data_v, ctp_lct_emul_v;
  ctp_lct_data_v.clear(); ctp_lct_emul_v.clear();
  if(ctp_lct_data_.isValid() && ctp_lct_emul_.isValid()) {
    int tbin_anode_offset = 5, emul_bx_corr;
    typedef CSCCorrelatedLCTDigiCollection::DigiRangeIterator mapIt;//map iterator
    typedef CSCCorrelatedLCTDigiCollection::const_iterator    vecIt;//vec iterator
    //loop over data (map<idx,vec_digi>)
    for (mapIt mit = ctp_lct_data_->begin(); mit != ctp_lct_data_->end(); mit++)
      //get vec_digi range(pair)  corresponding to idx of map
      //loop over digi vector (ie between begin and end pointers in range)
      //CSCCorrelatedLCTDigiCollection::Range ctpRange = ctp_lct_data_->get((*mit).first)
      //for (vecIt vit = ctpRange.first; vit != ctpRange.second; vit++) {
      for (vecIt vit = ctp_lct_data_->get((*mit).first).first; 
	   vit != ctp_lct_data_->get((*mit).first).second; vit++) 
	ctp_lct_data_v.push_back(*vit);
    for (mapIt mit = ctp_lct_emul_->begin(); mit != ctp_lct_emul_->end(); mit++) {
      const CSCDetId& detid = (*mit).first;

      // Extract full 12-bit BX word from ALCT data collections.
      int full_anode_bx = -999;
      if(ctp_ano_data_.isValid()) {
	const CSCALCTDigiCollection::Range& arange = ctp_ano_data_->get(detid);
	for (CSCALCTDigiCollection::const_iterator digiIt = arange.first;
	     digiIt != arange.second; digiIt++) {
	  if ((*digiIt).isValid()) {
	    full_anode_bx = (*digiIt).getFullBX();
	    break;
	  }
	}
      }

      for (vecIt vit = ctp_lct_emul_->get(detid).first; 
	   vit != ctp_lct_emul_->get(detid).second; vit++) {
	int emul_bx = (*vit).getBX();
	if (full_anode_bx != -999) {
	  emul_bx_corr = (full_anode_bx + emul_bx - tbin_anode_offset) & 0x01;
	}
	else { // This should never happen for default config. settings.
	  emul_bx_corr = emul_bx & 0x01;
	}

	// If one compares correlated LCTs after the muon port card, an
	// additional offset is needed.
	if (m_DEsource[CTP][1].instance() == "MPCSORTED") emul_bx_corr += 5;

	CSCCorrelatedLCTDigi lct((*vit).getTrknmb(),  (*vit).isValid(),
				 (*vit).getQuality(), (*vit).getKeyWG(),
				 (*vit).getStrip(),   (*vit).getPattern(),
				 (*vit).getBend(),    emul_bx_corr,
				 (*vit).getMPCLink(), (*vit).getBX0(),
				 (*vit).getSyncErr(), (*vit).getCSCID());
	ctp_lct_emul_v.push_back(lct);
      }
    }
  }
  ctp_lct_data =&ctp_lct_data_v;
  ctp_lct_emul =&ctp_lct_emul_v;


  // -- CTF [cathode strip chamber track finder]

  edm::Handle<L1MuRegionalCandCollection> ctf_data, ctf_emul;
  edm::Handle<L1CSCTrackCollection> ctf_trk_data_, ctf_trk_emul_; 

  CSCCorrelatedLCTDigiCollection_ const* ctf_trk_data(new CSCCorrelatedLCTDigiCollection_);
  CSCCorrelatedLCTDigiCollection_ const* ctf_trk_emul(new CSCCorrelatedLCTDigiCollection_);

  //L1MuRegionalCandCollection      const* ctf_trc_data(new L1MuRegionalCandCollection);
  //L1MuRegionalCandCollection      const* ctf_trc_emul(new L1MuRegionalCandCollection);

  edm::Handle<L1CSCStatusDigiCollection> ctf_sta_data_;
  edm::Handle<L1CSCStatusDigiCollection> ctf_sta_emul_;

  L1CSCSPStatusDigiCollection_    const* ctf_sta_data(new L1CSCSPStatusDigiCollection_);
  L1CSCSPStatusDigiCollection_    const* ctf_sta_emul(new L1CSCSPStatusDigiCollection_);

  if(m_doSys[CTF]) {
    iEvent.getByLabel(m_DEsource[CTF][2],ctf_trk_data_);
    iEvent.getByLabel(m_DEsource[CTF][3],ctf_trk_emul_);
    //note: unpacker different label: MounL1CSCTrackCollection
    iEvent.getByLabel(m_DEsource[CTF][0],ctf_data);
    iEvent.getByLabel(m_DEsource[CTF][1],ctf_emul);
    //note: unpacker only
    iEvent.getByLabel(m_DEsource[CTF][0].label(),"MuonL1CSCStatusDigiCollection",ctf_sta_data_);
    iEvent.getByLabel(m_DEsource[CTF][1].label(),"MuonL1CSCStatusDigiCollection",ctf_sta_emul_);
  }

  if(ctf_sta_data_.isValid())
    ctf_sta_data = &(ctf_sta_data_->second);

  if(ctf_sta_emul_.isValid())
    ctf_sta_emul = &(ctf_sta_emul_->second);

  CSCCorrelatedLCTDigiCollection_ ctf_trk_data_v, ctf_trk_emul_v; //vector
  L1MuRegionalCandCollection      ctf_trc_data_v, ctf_trc_emul_v; //vector

  if(ctf_trk_data_.isValid() && ctf_trk_emul_.isValid()) {
    typedef CSCCorrelatedLCTDigiCollection::DigiRangeIterator mapIt;//map iterator
    typedef CSCCorrelatedLCTDigiCollection::const_iterator    vecIt;//vec iterator
    typedef L1CSCTrackCollection::const_iterator ctcIt;

    //loop over csc-tracks (ie pairs<l1track,digi_vec>)
    for(ctcIt tcit=ctf_trk_data_->begin(); tcit!=ctf_trk_data_->end(); tcit++) {
      /// restrict comparison to middle of readout window
      if((tcit->first.bx() < -1) || (tcit->first.bx() > 1))
	continue;
      //store the muon candidate
      //csc::L1Track ttr = tcit->first;
      //L1MuRegionalCand cand(ttr);    
      //ctf_trc_data_v.push_back(tcit->first);
      ctf_trc_data_v.push_back(L1MuRegionalCand(tcit->first.getDataWord(), tcit->first.bx()));
      CSCCorrelatedLCTDigiCollection ldc = tcit->second; //muondigicollection=map
      //get the lct-digi-collection (ie muon-digi-collection)
      //loop over data (map<idx,vec_digi>)
      for (mapIt mit = ldc.begin(); mit != ldc.end(); mit++)
	//get vec_digi range(pair)  corresponding to idx of map
	//loop over digi vector (ie between begin and end pointers in range)
	//CSCCorrelatedLCTDigiCollection::Range ctpRange = ctp_lct_data_->get((*mit).first)
	//for (vecIt vit = ctpRange.first; vit != ctpRange.second; vit++) {
	for (vecIt vit = ldc.get((*mit).first).first; 
	     vit != ldc.get((*mit).first).second; vit++) 
	  ctf_trk_data_v.push_back(*vit);
    }

    //ctf_trk_data = &ctf_trk_data_v;
    //ctf_trc_data = &ctf_trc_data_v;

    //same for emulator collection
    for(ctcIt tcit=ctf_trk_emul_->begin();tcit!=ctf_trk_emul_->end(); tcit++) {
      if((tcit->first.bx() < -1) || (tcit->first.bx() > 1))
	continue;
      ctf_trc_emul_v.push_back(L1MuRegionalCand(tcit->first.getDataWord(), tcit->first.bx()));
      CSCCorrelatedLCTDigiCollection ldc = tcit->second;
      for (mapIt mit = ldc.begin(); mit != ldc.end(); mit++)
	for (vecIt vit = ldc.get((*mit).first).first; 
	     vit != ldc.get((*mit).first).second; vit++) 
	  ctf_trk_emul_v.push_back(*vit);
    }

    ctf_trk_emul = &ctf_trk_emul_v;
    //ctf_trc_emul = &ctf_trc_emul_v;

  }
  
  // -- RPC [resistive plate chambers regional trigger] 
  edm::Handle<L1MuRegionalCandCollection> rpc_cen_data;
  edm::Handle<L1MuRegionalCandCollection> rpc_cen_emul;
  edm::Handle<L1MuRegionalCandCollection> rpc_for_data;
  edm::Handle<L1MuRegionalCandCollection> rpc_for_emul;
  if(m_doSys[RPC]) {
    iEvent.getByLabel(m_DEsource[RPC][0].label(),"RPCb",rpc_cen_data);
    iEvent.getByLabel(m_DEsource[RPC][1].label(),"RPCb",rpc_cen_emul);
    iEvent.getByLabel(m_DEsource[RPC][0].label(),"RPCf",rpc_for_data);
    iEvent.getByLabel(m_DEsource[RPC][1].label(),"RPCf",rpc_for_emul);
  } 

  // -- LTC [local trigger controller]
  edm::Handle<LTCDigiCollection> ltc_data;
  edm::Handle<LTCDigiCollection> ltc_emul;
  if(m_doSys[LTC]) {
    iEvent.getByLabel(m_DEsource[LTC][0],ltc_data);
    iEvent.getByLabel(m_DEsource[LTC][1],ltc_emul);
  }

  // -- GMT [global muon trigger]
  edm::Handle<L1MuGMTCandCollection> gmt_data;
  edm::Handle<L1MuGMTCandCollection> gmt_emul;
  edm::Handle<L1MuGMTReadoutCollection> gmt_rdt_data_;
  edm::Handle<L1MuGMTReadoutCollection> gmt_rdt_emul_;
  L1MuRegionalCandCollection const* gmt_rdt_data(new L1MuRegionalCandCollection);
  L1MuRegionalCandCollection const* gmt_rdt_emul(new L1MuRegionalCandCollection);
  //tbd: may compare extended candidates
  L1MuGMTCandCollection const *gmt_can_data(new L1MuGMTCandCollection);
  L1MuGMTCandCollection const *gmt_can_emul(new L1MuGMTCandCollection);
  if(m_doSys[GMT]) {
    iEvent.getByLabel(m_DEsource[GMT][0], gmt_data);
    iEvent.getByLabel(m_DEsource[GMT][1], gmt_emul);
    iEvent.getByLabel(m_DEsource[GMT][0], gmt_rdt_data_);
    iEvent.getByLabel(m_DEsource[GMT][1], gmt_rdt_emul_);
  }  
  L1MuGMTCandCollection      gmt_can_data_vec, gmt_can_emul_vec;
  L1MuRegionalCandCollection gmt_rdt_data_vec, gmt_rdt_emul_vec;
  gmt_can_data_vec.clear();  gmt_can_emul_vec.clear();
  gmt_rdt_data_vec.clear();  gmt_rdt_emul_vec.clear();
  if( gmt_rdt_data_.isValid() && gmt_rdt_emul_.isValid() ) {
    typedef std::vector<L1MuGMTReadoutRecord>::const_iterator GmtRrIt;
    //get record vector for data 
    std::vector<L1MuGMTReadoutRecord> gmt_rdt_data_bx = gmt_rdt_data_->getRecords();
    for(GmtRrIt igmtrr=gmt_rdt_data_bx.begin(); igmtrr!=gmt_rdt_data_bx.end(); igmtrr++) {
      //get gmt cands
      typedef std::vector<L1MuGMTExtendedCand>::const_iterator GmtECIt;
      std::vector<L1MuGMTExtendedCand> gmc;
      gmc = igmtrr->getGMTCands();
      for(GmtECIt iter1=gmc.begin(); iter1!=gmc.end(); iter1++) {
	L1MuGMTCand cand(iter1->getDataWord(),iter1->bx());
	cand.setPhiValue(iter1->phiValue());
	cand.setEtaValue(iter1->etaValue());
	cand.setPtValue (iter1->ptValue ());
	gmt_can_data_vec.push_back(cand);
      }
      //get reg cands
      typedef L1MuRegionalCandCollection::const_iterator GmtRCIt;
      L1MuRegionalCandCollection rmc;
      rmc.clear();
      rmc = igmtrr->getDTBXCands();
      gmt_rdt_data_vec.insert(gmt_rdt_data_vec.end(),rmc.begin(),rmc.end());
      rmc.clear();
      rmc = igmtrr->getCSCCands();
      gmt_rdt_data_vec.insert(gmt_rdt_data_vec.end(),rmc.begin(),rmc.end());
      rmc.clear();
      rmc = igmtrr->getBrlRPCCands();
      gmt_rdt_data_vec.insert(gmt_rdt_data_vec.end(),rmc.begin(),rmc.end());
      rmc.clear();
      rmc = igmtrr->getFwdRPCCands();
      gmt_rdt_data_vec.insert(gmt_rdt_data_vec.end(),rmc.begin(),rmc.end());
    }
    //get record vector for emul 
    std::vector<L1MuGMTReadoutRecord> gmt_rdt_emul_bx = gmt_rdt_emul_->getRecords();
    for(GmtRrIt igmtrr=gmt_rdt_emul_bx.begin(); igmtrr!=gmt_rdt_emul_bx.end(); igmtrr++) {
      //get gmt cands
      typedef std::vector<L1MuGMTExtendedCand>::const_iterator GmtECIt;
      std::vector<L1MuGMTExtendedCand> gmc;
      gmc = igmtrr->getGMTCands();
      for(GmtECIt iter1=gmc.begin(); iter1!=gmc.end(); iter1++) {
	gmt_can_emul_vec.push_back(L1MuGMTCand(iter1->getDataWord(),iter1->bx()));
      }
      //get reg cands
      typedef L1MuRegionalCandCollection::const_iterator GmtRCIt;
      L1MuRegionalCandCollection rmc;
      rmc.clear();
      rmc = igmtrr->getDTBXCands();
      gmt_rdt_emul_vec.insert(gmt_rdt_emul_vec.end(),rmc.begin(),rmc.end());
      rmc.clear();
      rmc = igmtrr->getCSCCands();
      gmt_rdt_emul_vec.insert(gmt_rdt_emul_vec.end(),rmc.begin(),rmc.end());
      rmc.clear();
      rmc = igmtrr->getBrlRPCCands();
      gmt_rdt_emul_vec.insert(gmt_rdt_emul_vec.end(),rmc.begin(),rmc.end());
      rmc.clear();
      rmc = igmtrr->getFwdRPCCands();
      gmt_rdt_emul_vec.insert(gmt_rdt_emul_vec.end(),rmc.begin(),rmc.end());
    }
  }
  gmt_rdt_data = &gmt_rdt_data_vec;
  gmt_rdt_emul = &gmt_rdt_emul_vec;
  gmt_can_data = &gmt_can_data_vec;
  gmt_can_emul = &gmt_can_emul_vec;

  // -- GLT [global trigger]
  edm::Handle<L1GlobalTriggerReadoutRecord>    glt_rdt_data;
  edm::Handle<L1GlobalTriggerReadoutRecord>    glt_rdt_emul;
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> glt_evm_data;
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> glt_evm_emul;
  edm::Handle<L1GlobalTriggerObjectMapRecord>  glt_obj_data;
  edm::Handle<L1GlobalTriggerObjectMapRecord>  glt_obj_emul;
  if(m_doSys[GLT]) {
    iEvent.getByLabel(m_DEsource[GLT][0], glt_rdt_data);
    iEvent.getByLabel(m_DEsource[GLT][1], glt_rdt_emul);
    iEvent.getByLabel(m_DEsource[GLT][0], glt_evm_data);
    iEvent.getByLabel(m_DEsource[GLT][1], glt_evm_emul);
    iEvent.getByLabel(m_DEsource[GLT][0], glt_obj_data);
    iEvent.getByLabel(m_DEsource[GLT][1], glt_obj_emul);
  }
  
  ///--- done getting collections. --- 

  //check collections validity
  bool isValidDE[DEnsys][2];// = {false};
  for(int i=0; i<DEnsys; i++) for(int j=0; j<2; j++) isValidDE[i][j]=false;

  isValidDE[ETP][0] =     ecal_tp_data .isValid(); isValidDE[ETP][1] =    ecal_tp_emul .isValid();
  isValidDE[HTP][0] =     hcal_tp_data .isValid(); isValidDE[HTP][1] =    hcal_tp_emul .isValid();
  isValidDE[RCT][0] =      rct_em_data .isValid(); isValidDE[RCT][1] =     rct_em_emul .isValid();
  isValidDE[RCT][0]&=     rct_rgn_data .isValid(); isValidDE[RCT][1] =    rct_rgn_emul .isValid();
  isValidDE[GCT][0] = gct_isolaem_data .isValid(); isValidDE[GCT][1] =gct_isolaem_emul .isValid();
  isValidDE[GCT][0]&= gct_noisoem_data .isValid(); isValidDE[GCT][1]&=gct_noisoem_emul .isValid();
  isValidDE[GCT][0]&= gct_cenjets_data .isValid(); isValidDE[GCT][1]&=gct_cenjets_emul .isValid();
  isValidDE[GCT][0]&= gct_forjets_data .isValid(); isValidDE[GCT][1]&=gct_forjets_emul .isValid();
  isValidDE[GCT][0]&= gct_taujets_data .isValid(); isValidDE[GCT][1]&=gct_taujets_emul .isValid();
  isValidDE[GCT][0]&=  gct_etmiss_data .isValid(); isValidDE[GCT][1]&= gct_etmiss_emul .isValid();
  isValidDE[GCT][0]&=  gct_ettota_data .isValid(); isValidDE[GCT][1]&= gct_ettota_emul .isValid();
  isValidDE[GCT][0]&=  gct_htmiss_data .isValid(); isValidDE[GCT][1]&= gct_htmiss_emul .isValid();
  isValidDE[GCT][0]&=  gct_hfring_data .isValid(); isValidDE[GCT][1]&= gct_hfring_emul .isValid();
  isValidDE[GCT][0]&=  gct_hfbcnt_data .isValid(); isValidDE[GCT][1]&= gct_hfbcnt_emul .isValid();
//isValidDE[GCT][0]&=  gct_jetcnt_data .isValid(); isValidDE[GCT][1]&= gct_jetcnt_emul .isValid(); #temporary
  isValidDE[DTP][0] =      dtp_ph_data_.isValid(); isValidDE[DTP][1] =     dtp_ph_emul_.isValid();
  isValidDE[DTP][0]&=      dtp_th_data_.isValid(); isValidDE[DTP][1]&=     dtp_th_emul_.isValid();
  isValidDE[DTF][0] =     dtf_trk_data_.isValid(); isValidDE[DTF][1] =    dtf_trk_emul_.isValid();
//isValidDE[DTF][0]&=         dtf_data .isValid(); isValidDE[DTF][1]&=        dtf_emul .isValid();
  isValidDE[CTP][0] =     ctp_lct_data_.isValid(); isValidDE[CTP][1] =    ctp_lct_emul_.isValid();
  if (m_DEsource[CTP][0].label().find("tf") == std::string::npos) {
  isValidDE[CTP][0]&=     ctp_ano_data_.isValid(); isValidDE[CTP][1]&=    ctp_ano_emul_.isValid();
  isValidDE[CTP][0]&=     ctp_cat_data_.isValid(); isValidDE[CTP][1]&=    ctp_cat_emul_.isValid();
  }
  isValidDE[CTF][0] =         ctf_data .isValid(); isValidDE[CTF][1] =        ctf_emul .isValid();
  isValidDE[CTF][0]&=    ctf_trk_data_ .isValid(); isValidDE[CTF][1]&=   ctf_trk_emul_ .isValid();
  //isValidDE[CTF][0]&=    ctf_sta_data_ .isValid(); isValidDE[CTF][1]&=   ctf_sta_emul_ .isValid();
  isValidDE[RPC][0] =     rpc_cen_data .isValid(); isValidDE[RPC][1] =    rpc_cen_emul .isValid();
  isValidDE[RPC][0]&=     rpc_for_data .isValid(); isValidDE[RPC][1]&=    rpc_for_emul .isValid();
  isValidDE[LTC][0] =         ltc_data .isValid(); isValidDE[LTC][1] =        ltc_emul .isValid();
  isValidDE[GMT][0] =         gmt_data .isValid(); isValidDE[GMT][1] =        gmt_emul .isValid();
//isValidDE[GMT][0]&=     gmt_rdt_data_.isValid(); isValidDE[GMT][1]&=    gmt_rdt_emul_.isValid();
  isValidDE[GLT][0] =     glt_rdt_data .isValid(); isValidDE[GLT][1] =    glt_rdt_emul .isValid();
//isValidDE[GLT][0]&=     glt_evm_data .isValid(); isValidDE[GLT][1]&=    glt_evm_emul .isValid();
//isValidDE[GLT][0]&=     glt_obj_data .isValid(); isValidDE[GLT][1]&=    glt_obj_emul .isValid();

  bool isValid[DEnsys];
  for(int i=0; i<DEnsys; i++) {
    isValid[i]=true;
    for(int j=0; j<2; j++) {
      isValid[i] &= isValidDE[i][j];
    }
  }

  if(verbose()) {
    std::cout << "L1Comparator sys isValid?  (evt:" << nevt_ << ") ";
    std::cout << "\n\t&: ";
    for(int i=0; i<DEnsys; i++)
      std::cout << isValid[i] << " ";
    std::cout << "\n\td: ";
    for(int i=0; i<DEnsys; i++)
      std::cout << isValidDE[i][0] << " ";
    std::cout << "\n\te: ";
    for(int i=0; i<DEnsys; i++)
      std::cout << isValidDE[i][1] << " ";
    std::cout << std::endl;
  }
  
  //reset flags...
  //for(int i=0; i<DEnsys; i++) isValid[i]=true;

  if(verbose())
    std::cout << "L1Comparator start processing the collections.\n" << std::flush;

  ///processing : compare the pairs of collections 
  if(m_doSys[ETP]&&isValid[ETP]) process<EcalTrigPrimDigiCollection>     (    ecal_tp_data,     ecal_tp_emul, ETP,ECALtp);
  if(m_doSys[HTP]&&isValid[HTP]) process<HcalTrigPrimDigiCollection>     (    hcal_tp_data,     hcal_tp_emul, HTP,HCALtp);
  if(m_doSys[RCT]&&isValid[RCT]) process<L1CaloEmCollection>             (     rct_em_data,      rct_em_emul, RCT,RCTem);
  if(m_doSys[RCT]&&isValid[RCT]) process<L1CaloRegionCollection>         (    rct_rgn_data,     rct_rgn_emul, RCT,RCTrgn);
  if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEmCandCollection>          (gct_isolaem_data, gct_isolaem_emul, GCT,GCTisolaem);
  if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEmCandCollection>          (gct_noisoem_data, gct_noisoem_emul, GCT,GCTnoisoem);
  if(m_doSys[GCT]&&isValid[GCT]) process<L1GctJetCandCollection>         (gct_cenjets_data, gct_cenjets_emul, GCT,GCTcenjets);
  if(m_doSys[GCT]&&isValid[GCT]) process<L1GctJetCandCollection>         (gct_forjets_data, gct_forjets_emul, GCT,GCTforjets);
  if(m_doSys[GCT]&&isValid[GCT]) process<L1GctJetCandCollection>         (gct_taujets_data, gct_taujets_emul, GCT,GCTtaujets);
  if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEtHadCollection>	         (     gct_ht_data,      gct_ht_emul, GCT,GCTethad);
  if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEtMissCollection>	         ( gct_etmiss_data,  gct_etmiss_emul, GCT,GCTetmiss);
  if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEtTotalCollection>	 ( gct_ettota_data , gct_ettota_emul, GCT,GCTettot);
  if(m_doSys[GCT]&&isValid[GCT]) process<L1GctHtMissCollection>	         ( gct_htmiss_data,  gct_htmiss_emul, GCT,GCThtmiss);
  if(m_doSys[GCT]&&isValid[GCT]) process<L1GctHFRingEtSumsCollection>	 ( gct_hfring_data,  gct_hfring_emul, GCT,GCThfring);
  if(m_doSys[GCT]&&isValid[GCT]) process<L1GctHFBitCountsCollection>	 ( gct_hfbcnt_data,  gct_hfbcnt_emul, GCT,GCThfbit);
//if(m_doSys[GCT]&&isValid[GCT]) process<L1GctJetCountsCollection>	 ( gct_jetcnt_data,  gct_jetcnt_emul, GCT,GCTjetcnt);#missing in emulator
  if(m_doSys[DTP]&&isValid[DTP]) process<L1MuDTChambPhDigiCollection>    (     dtp_ph_data,      dtp_ph_emul, DTP,DTtpPh);
  if(m_doSys[DTP]&&isValid[DTP]) process<L1MuDTChambThDigiCollection>    (     dtp_th_data,      dtp_th_emul, DTP,DTtpTh);
  if(m_doSys[DTF]&&isValid[DTF]) process<L1MuRegionalCandCollection>     (        dtf_data,         dtf_emul, DTF,DTtf);
  if(m_doSys[DTF]&&isValid[DTF]) process<L1MuRegionalCandCollection>     (    dtf_trk_data,     dtf_trk_emul, DTF,DTtftrk);
  if(m_DEsource[CTP][0].label().find("tf") == std::string::npos) {
  if(m_doSys[CTP]&&isValid[CTP]) process<CSCALCTDigiCollection_>         (    ctp_ano_data,     ctp_ano_emul, CTP,CSCtpa);
  if(m_doSys[CTP]&&isValid[CTP]) process<CSCCLCTDigiCollection_>         (    ctp_cat_data,     ctp_cat_emul, CTP,CSCtpc);
  }
  if(m_doSys[CTP]&&isValid[CTP]) process<CSCCorrelatedLCTDigiCollection_>(    ctp_lct_data,     ctp_lct_emul, CTP,CSCtpl);
  if(m_doSys[CTF]&&isValid[CTF]) process<L1MuRegionalCandCollection>     (        ctf_data,         ctf_emul, CTF,CSCtf);
  if(m_doSys[CTF]&&isValid[CTF]) process<CSCCorrelatedLCTDigiCollection_>(    ctf_trk_data,     ctf_trk_emul, CTF,CSCtftrk);
  //if(m_doSys[CTF]&&isValid[CTF]) process<L1MuRegionalCandCollection>     (    ctf_trc_data,     ctf_trc_emul, CTF,CSCtftrc);
  if(m_doSys[CTF]&&isValid[CTF]) process<L1CSCSPStatusDigiCollection_>   (    ctf_sta_data,     ctf_sta_emul, CTF,CSCtfsta);
  if(m_doSys[RPC]&&isValid[RPC]) process<L1MuRegionalCandCollection>     (    rpc_cen_data,     rpc_cen_emul, RPC,RPCcen);
  if(m_doSys[RPC]&&isValid[RPC]) process<L1MuRegionalCandCollection>     (    rpc_for_data,     rpc_for_emul, RPC,RPCfor);
  if(m_doSys[LTC]&&isValid[LTC]) process<LTCDigiCollection>              (        ltc_data,         ltc_emul, LTC,LTCi);
  if(m_doSys[GMT]&&isValid[GMT]) process<L1MuGMTCandCollection>          (        gmt_data,         gmt_emul, GMT,GMTmain);
  if(m_doSys[GMT]&&isValid[GMT]) process<L1MuRegionalCandCollection>     (    gmt_rdt_data,     gmt_rdt_emul, GMT,GMTrdt);
  if(m_doSys[GMT]&&isValid[GMT]) process<L1MuGMTCandCollection>          (    gmt_can_data,     gmt_can_emul, GMT,GMTcnd);

  // >>---- GLT ---- <<  
  GltDEDigi gltdigimon;
  
  if(m_doSys[GLT] && isValid[GLT] ) {

    ///tmp: for getting a clean dump (avoid empty entries)
    bool prt = false; 
    if(!m_dumpMode)
      prt = false;
    else if(m_dumpMode==-1)
      prt=true;

    if(dumpEvent_ && prt) {
      m_dumpFile << "\nEntry: " << nevt_ 
		 << " (event:"  << evtNum_
		 << " | run:"   << runNum_ 
		 << ")\n"       << std::flush;
      dumpEvent_=false;
    }

    m_dumpFile << "\n  GT...\n";

    if(glt_rdt_data.isValid() && glt_rdt_emul.isValid()) {
      
      //fill gt mon info
      bool globalDBit[2];
      std::vector<bool> gltDecBits[2], gltTchBits[2];
      globalDBit[0] = glt_rdt_data->decision();
      globalDBit[1] = glt_rdt_emul->decision();
      gltDecBits[0] = glt_rdt_data->decisionWord();
      gltDecBits[1] = glt_rdt_emul->decisionWord();
      //gltTchBits[0] = glt_rdt_data->gtFdlWord().gtTechnicalTriggerWord();
      //gltTchBits[1] = glt_rdt_emul->gtFdlWord().gtTechnicalTriggerWord();
      gltTchBits[0] = glt_rdt_data->technicalTriggerWord();
      gltTchBits[1] = glt_rdt_emul->technicalTriggerWord();
      gltdigimon.set(globalDBit, gltDecBits, gltTchBits);

      DEncand[GLT][0]=1; DEncand[GLT][1]=1;
      DEmatchEvt[GLT]  = compareCollections(glt_rdt_data, glt_rdt_emul);  
    }

    /// (may skip further collection checks temporarily...)
    if(glt_evm_data.isValid() && glt_evm_emul.isValid())
      DEmatchEvt[GLT] &= compareCollections(glt_evm_data, glt_evm_emul);  
    if(glt_obj_data.isValid() && glt_obj_emul.isValid())
      DEmatchEvt[GLT] &= compareCollections(glt_obj_data, glt_obj_emul);  

    char ok[10];
    char dumptofile[1000];
    if(DEmatchEvt[GLT]) sprintf(ok,"successful");
    else         sprintf(ok,"failed");
    sprintf(dumptofile,"  ...GT data and emulator comparison: %s\n", ok); 
    m_dumpFile<<dumptofile;
  }

  if(verbose())
    std::cout << "L1Comparator done processing all collections.\n" << std::flush;

  if(verbose()) {
    std::cout << "[L1Comparator] sys match? << evt." << nevt_ << ": ";
    for(int i=0; i<DEnsys; i++)
      std::cout << DEmatchEvt[i] << " ";
    std::cout << std::endl;
  }


  // >>---- Event match? ---- <<  

  bool evt_match  = true;
  for(int i=0; i<DEnsys; i++)
    evt_match &= DEmatchEvt[i];

  
  /* char ok[10];
     if(evt_match) sprintf(ok,"GOOD :]");
     else      sprintf(ok,"BAD !!!"); 
     char dumptofile[1000];
     sprintf(dumptofile,"\n -> event data and emulator match... %s\n", ok);
     m_dumpFile<<dumptofile;
  */

  // >>---- Global match? ---- <<  
  m_match &= evt_match;
  m_dumpFile << std::flush;

  //if collection is empty, add empty digi
  if(m_dedigis.size()==0) {
    if(verbose())
      std::cout << "\n [L1Comparator] adding empty collection to DErecord\n";
    m_dedigis.push_back(L1DataEmulDigi());
  }
  
  // >>---- d|e record ---- <<  
  std::auto_ptr<L1DataEmulRecord> record
    (new L1DataEmulRecord(evt_match,m_doSys,DEmatchEvt,DEncand,m_dedigis, gltdigimon));
  if(verbose()) {
    std::cout << "\n [L1Comparator] printing DErecord" 
	      << "(entry:"<< nevt_ 
	      << "|evt:"  << evtNum_
	      << "|run:"  << runNum_ 
	      << "):\n"    << std::flush;
    std::cout << *record 
	      << "\n" << std::flush;
  }

  iEvent.put(record);

  /// further analysis
  bool dorawdata = false;
  if(dorawdata) {
    edm::Handle<FEDRawDataCollection> raw_fedcoll_data;
    edm::Handle<FEDRawDataCollection> raw_fedcoll_emul;
    iEvent.getByLabel(m_FEDsource[0], raw_fedcoll_data);
    iEvent.getByLabel(m_FEDsource[1], raw_fedcoll_emul);
    bool rawval=true;
    rawval &= raw_fedcoll_data.isValid();
    rawval &= raw_fedcoll_emul.isValid();
    if(rawval) 
      compareFedRawCollections(raw_fedcoll_data,raw_fedcoll_emul, m_fedId);
  }
  
  if(verbose())
    std::cout << "L1comparator::analize() end. " << nevt_ << std::endl;
  
}


template <class T> 
void L1Comparator::process(T const* data, T const* emul, const int sys, const int cid) {

  if(verbose())
    std::cout << "L1Comparator::process -ing system:" << sys 
	      << " (" << SystLabel[sys] << "), data type " << cid 
	      << "...\n" << std::flush;
  if(verbose())
  std::cout << "L1Comparator::process debug "
	    << " (size "  << data->size() << "," <<emul->size() << ")"   
	    << ".\n" << std::flush;

  ///tmp: for getting a clean dump (avoid empty entries)
  bool prt = false; 
  if(!m_dumpMode)
    prt = false;
  else if(m_dumpMode==-1)
    prt=true;
  else if(m_dumpMode>0) {
    DEcompare<T> tmp(data,emul);
    if(tmp.get_ncand(0)==0 && tmp.get_ncand(1)==0)
      prt=false;
    else
    prt = !tmp.do_compare(m_dumpFile,0);
  }

  //declare de compare object
  DEcompare<T> cmp(data,emul);

  int ndata = cmp.get_ncand(0);
  int nemul = cmp.get_ncand(1);
  
  if(verbose())
    std::cout << "L1Comparator::process " 
	      << " system:" << SystLabel[sys] << "(id " << sys << ")" 
	      << " type:"   << cmp.GetName(0) << "(" << cmp.de_type() << ")"
	      << " ndata:"  << ndata
	      << " nemul:"  << nemul
	      << " (size "  << data->size() << "," <<emul->size() << ")"   
	      << ".\n" << std::flush;
  
  if(ndata==0&&nemul==0) {
    if(verbose())
      std::cout << "L1Comparator::process " 
		<< "empty collections -- exiting!\n" << std::flush;
    return;
  }
  
  m_dumpFile << std::setiosflags(std::ios::showpoint | std::ios::fixed 
				 | std::ios::right | std::ios::adjustfield);
  std::cout  << std::setiosflags(std::ios::showpoint | std::ios::fixed 
				 | std::ios::right | std::ios::adjustfield);
  
  if(dumpEvent_ &&  prt ) {
    m_dumpFile << "\nEntry: " << nevt_ 
	       << " (event:"  << evtNum_
	       << " | run:"   << runNum_ 
	       << ")\n"       << std::flush;
    dumpEvent_=false;
  }
  
  if(prt)
    m_dumpFile << "\n  sys:" << SystLabel[sys] 
	       << " (" << sys << "), type:" << cid //cmp.GetName() 
      	       << " ...\n";
  
  if(verbose())
    std::cout << "L1Comparator::process print:\n" << std::flush
	      << cmp.print()
	      << std::flush;
  
  ///perform comparison
  DEmatchEvt[sys] &= cmp.do_compare(m_dumpFile,m_dumpMode);
  
  ///gather results
  L1DEDigiCollection dg = cmp.getDEDigis();

  if(verbose())
    for(L1DEDigiCollection::iterator it=dg.begin(); it!=dg.end();it++)
      std::cout << *it << "\n";

  ///over-write system-id: needed eg for GMT input, CSC tf reg cand, CTP&CTF
  for(L1DEDigiCollection::iterator it=dg.begin(); it!=dg.end();it++)
    it->setSid(sys);
  ///over-write data type: needed eg for GCT jet types, regional muon sources
  for(L1DEDigiCollection::iterator it=dg.begin(); it!=dg.end();it++)
    it->setCid(cid);

  ///append d|e digis to the record's collection
  m_dedigis.insert(m_dedigis.end(), dg.begin(), dg.end()); 
  for(int i=0; i<2; i++)
    DEncand[sys][i] += cmp.get_ncand(i);

  if(verbose())
    std::cout << "L1Comparator::process " 
	      << " system:" << SystLabel[sys] 
	      << " type:"   << cmp.GetName(0) 
	      << " ndata:"  << DEncand[sys][0]
	      << " nemul:"  << DEncand[sys][1]
	      << " (size "  << data->size() << "," <<emul->size() << ")"   
	      << " ndigis:" << dg.size()
	      << " agree? " << DEmatchEvt[sys]
	      << std::endl;

  if(verbose())
    std::cout << "L1Comparator::process -ing system:" 
	      << sys << " (" << SystLabel[sys] << ")...done.\n" 
	      << std::flush;
}

//--//--//--//--//--//--//--//--//--//--//--//--//--//--//--//--

bool
L1Comparator::compareCollections(edm::Handle<L1GlobalTriggerReadoutRecord> data, 
				 edm::Handle<L1GlobalTriggerReadoutRecord> emul) {

  if(verbose())
    std::cout << "L1Comparator -- result of GT embedded comparison.\n"
	      << "L1GlobalTriggerReadoutRecord:: data and emulator agree? "
	      << ((*data==*emul)?"yes":"no")
	      << std::endl;
  
  m_dumpFile << "\n L1GlobalTriggerReadoutRecord candidates...\n";

  bool thematch = true;
  
  thematch &= (*data==*emul);
  
  bool match = thematch;

  if(m_dumpMode==0 && match)
    return match;    
  
  //expand to check mismatching  stage

  //need to create new objects due to lack of suitable accessors
  // needed only for accessing gtPsbVector()
  std::auto_ptr<L1GlobalTriggerReadoutRecord> 
    data_( new L1GlobalTriggerReadoutRecord(*(data.product())));
  std::auto_ptr<L1GlobalTriggerReadoutRecord> 
    emul_( new L1GlobalTriggerReadoutRecord(*(emul.product())));

  match = true;
  m_dumpFile << "\tmatch stage: ";
  match &= (data->gtfeWord()            == emul->gtfeWord()     );
  m_dumpFile << " gtfeWord:" << match; 	        
  match &= (data->gtFdlWord()           == emul->gtFdlWord()    );
  m_dumpFile << " gtFdlWord:" << match; 	        
  match &= (data->muCollectionRefProd() == emul->muCollectionRefProd());
  m_dumpFile << " muCollectionRefProd:" << match << "\n"; 
  boost::uint16_t dt_psb_bid=0, em_psb_bid=0;    
  size_t npsbw = (data_->gtPsbVector().size()>emul_->gtPsbVector().size())?
    emul_->gtPsbVector().size():data_->gtPsbVector().size();
  for(int idx=0; idx<(int)npsbw; idx++) {
    if(data_->gtPsbVector().at(idx) != emul_->gtPsbVector().at(idx) ) {
      //match &= false;
      dt_psb_bid = data_->gtPsbVector().at(idx).boardId();
      em_psb_bid = emul_->gtPsbVector().at(idx).boardId();
      break;
    }
  }
  match &= (data->gtPsbWord(dt_psb_bid) == emul->gtPsbWord(em_psb_bid) );
  //if(!match) {
  //  m_dumpFile << "  data"; data_->gtPsbWord(dt_psb_bid).print(m_dumpFile);
  //  m_dumpFile << "\nemul"; emul_->gtPsbWord(em_psb_bid).print(m_dumpFile);
  //}
  //problem: vector not accessible from handle (only reference non-const)
  //std::vector<L1GtPsbWord>& data_psbVec = data_->gtPsbVector();
  //std::vector<L1GtPsbWord>& emul_psbVec = emul_->gtPsbVector();
  m_dumpFile << " gtPsbWord("<<dt_psb_bid<<","<<em_psb_bid<<"):" << match << "\n"; 

  ///todo: skip empty events

  // gt decision
  m_dumpFile << "\n\tGlobal decision: "
    	   << data->decision() << " (data) "
    	   << emul->decision() << " (emul) "
	   << std::endl;

  // gt decision word
  m_dumpFile << "\n\tDecisionWord  (bits: 63:0, 127:64)";
  int nbitword = 64; 
  std::vector<bool> data_gtword = data->decisionWord();
  std::vector<bool> emul_gtword = emul->decisionWord();
  m_dumpFile << "\n\tdata: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (data_gtword.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << "\n\t      ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (data_gtword.at(nbitword*2-1-i)  ? '1' : '0');
  }
  m_dumpFile << "\n\temul: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (emul_gtword.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << "\n\t      ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (emul_gtword.at(nbitword*2-1-i)  ? '1' : '0');
  }
  m_dumpFile << std::endl;
 
  m_dumpFile << "\n\tDecisionWordExtended  (bits: 0:63)";
  std::vector<bool> data_decwext = data->gtFdlWord().gtDecisionWordExtended();
  std::vector<bool> emul_decwext = emul->gtFdlWord().gtDecisionWordExtended();
  m_dumpFile << "\n\tdata: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (data_decwext.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << "\n\temul: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (emul_decwext.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << std::endl;

  m_dumpFile << "\n\tTechnical triggers (bits: 0:63)";
  std::vector<bool> data_fdlttw = data->gtFdlWord().gtTechnicalTriggerWord();
  std::vector<bool> emul_fdlttw = emul->gtFdlWord().gtTechnicalTriggerWord();
  assert((int)data_fdlttw.size()==nbitword); 
  m_dumpFile << "\n\tdata: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (data_fdlttw.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << "\n\temul: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (emul_fdlttw.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << std::endl;

  m_dumpFile << "\n\tL1GtFdlWord";
  m_dumpFile << "\n\tdata: "
	   << " BoardId:"   << data->gtFdlWord().boardId()
	   << " BxInEvent:" << data->gtFdlWord().bxInEvent()
	   << " BxNr:"      << data->gtFdlWord().bxNr()
	   << " EventNr:"   << data->gtFdlWord().eventNr()
	   << " NoAlgo:"    << data->gtFdlWord().noAlgo()
	   << " FinalOR:"   << data->gtFdlWord().finalOR()
	   << " LocalBxNr:" << data->gtFdlWord().localBxNr();
  m_dumpFile << "\n\temul: "
	   << " BoardId:"   << emul->gtFdlWord().boardId()
	   << " BxInEvent:" << emul->gtFdlWord().bxInEvent()
	   << " BxNr:"      << emul->gtFdlWord().bxNr()
	   << " EventNr:"   << emul->gtFdlWord().eventNr()
	   << " NoAlgo:"    << emul->gtFdlWord().noAlgo()
	   << " FinalOR:"   << emul->gtFdlWord().finalOR()
	   << " LocalBxNr:" << emul->gtFdlWord().localBxNr()
	   << std::endl;

  m_dumpFile << "\n\tL1GtfeWord";
  m_dumpFile << "\n\tdata: " 
	   << " BoardId:"        << data->gtfeWord().boardId()
	   << " RecordLength:"   << data->gtfeWord().recordLength()
	   << " BxNr:"           << data->gtfeWord().bxNr() 
	   << " SetupVersion:"   << data->gtfeWord().setupVersion()
	   << " ActiveBoards:"   << data->gtfeWord().activeBoards()
	   << " TotalTriggerNr:" << data->gtfeWord().totalTriggerNr();
  m_dumpFile << "\n\temul: " 
	   << " BoardId:"        << emul->gtfeWord().boardId()
	   << " RecordLength:"   << emul->gtfeWord().recordLength()
	   << " BxNr:"           << emul->gtfeWord().bxNr() 
	   << " SetupVersion:"   << emul->gtfeWord().setupVersion()
	   << " ActiveBoards:"   << emul->gtfeWord().activeBoards()
	   << " TotalTriggerNr:" << emul->gtfeWord().totalTriggerNr()
	   << std::endl;

  //uint16_t psb_bid = (uint16_t)idx; //need to find relevant board-id to dump
  m_dumpFile << "\n\tgtPsbWord";
  m_dumpFile << "\n\tdata: "
	   << " Board Id:"  << data->gtPsbWord(dt_psb_bid).boardId()
	   << " BxInEvent:" << data->gtPsbWord(dt_psb_bid).bxInEvent()
	   << " BxNr:"      << data->gtPsbWord(dt_psb_bid).bxNr()
	   << " LocalBxNr:" << data->gtPsbWord(dt_psb_bid).localBxNr()
	   << " EventNr:"   << data->gtPsbWord(dt_psb_bid).eventNr();
  m_dumpFile << "\n\temul: "
	   << " Board Id:"  << emul->gtPsbWord(em_psb_bid).boardId()
	   << " BxInEvent:" << emul->gtPsbWord(em_psb_bid).bxInEvent()
	   << " BxNr:"      << emul->gtPsbWord(em_psb_bid).bxNr()
	   << " LocalBxNr:" << emul->gtPsbWord(em_psb_bid).localBxNr()
	   << " EventNr:"   << emul->gtPsbWord(em_psb_bid).eventNr()
	   << std::endl;
  
  // m_dumpFile << "\n\tA,B_Data_CH7:0"
  //	   << " ...waiting for data accessors in dataformats!\n\n";
  //#include "DataFormats/L1GlobalTrigger/src/L1GtPsbWord.cc"
  
  m_dumpFile << "\n\tA_Data_CH7:0";
  m_dumpFile << "\n\tdata: ";
  for (int i=0; i<8; ++i)
    m_dumpFile << data->gtPsbWord(dt_psb_bid).aData(7-i) << " ";
  m_dumpFile << "\n\temul: ";
  for (int i=0; i<8; ++i)
    m_dumpFile << emul->gtPsbWord(em_psb_bid).aData(7-i) << " ";
  m_dumpFile << std::endl;
  
  m_dumpFile << "\n\tA_Data_CH7:0";
  m_dumpFile << "\n\tdata: ";
  for (int i=0; i<8; ++i)
    m_dumpFile << data->gtPsbWord(dt_psb_bid).bData(7-i) << " ";
  m_dumpFile << "\n\temul: ";
  for (int i=0; i<8; ++i)
    m_dumpFile << emul->gtPsbWord(em_psb_bid).bData(7-i) << " ";
  m_dumpFile << "\n" << std::endl;


  /// todo printL1Objects!

  /// debug: print it all ()
  if(false) {
    m_dumpFile << "---debug: print full gt record---";
    m_dumpFile << "\n\tdata: ";   
    data->print(m_dumpFile);
    m_dumpFile << "\n\temul: ";   
    emul->print(m_dumpFile);
    m_dumpFile << "\n"; 
    m_dumpFile << "---debug: print full gt record Done.---\n\n";
  }

  char ok[10];
  if(match) sprintf(ok,"successful");
  else      sprintf(ok,"failed");
  m_dumpFile << " ...L1GlobalTriggerReadoutRecord data and emulator comparison: " 
	   << ok << std::endl;

  return thematch;
}


bool
L1Comparator::compareCollections(edm::Handle<L1GlobalTriggerEvmReadoutRecord> data, 
				 edm::Handle<L1GlobalTriggerEvmReadoutRecord> emul) {

  if(verbose())
    std::cout << "L1Comparator -- result of GT embedded comparison.\n"
	      << "L1GlobalTriggerEvmReadoutRecord data and emulator agree? "
	      << ((*data==*emul)?"yes":"no")
	      << std::endl;
  
  m_dumpFile << "\n  L1GlobalTriggerEvmReadoutRecord candidates...\n";
  
  bool match = true;
  match &= (*data==*emul);
  
  if(m_dumpMode==0 && match)
    return match;
  
  // gt decision
  m_dumpFile << "\n\tGlobal decision: "
	     << data->decision() << " (data) "
	     << emul->decision() << " (emul) "
	     << std::endl;

  // gt decision word
  m_dumpFile << "\n\tDecisionWord  (bits: 0:63, 127:64)";
  int nbitword = 64; 
  std::vector<bool> data_gtword = data->decisionWord();
  std::vector<bool> emul_gtword = emul->decisionWord();
  m_dumpFile << "\n\tdata: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (data_gtword.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << "\n\t      ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (data_gtword.at(nbitword*2-1-i)  ? '1' : '0');
  }
  m_dumpFile << "\n\temul: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (emul_gtword.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << "\n\t      ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (emul_gtword.at(nbitword*2-1-i)  ? '1' : '0');
  }
  m_dumpFile << std::endl;

  m_dumpFile << "\n\tDecisionWordExtended  (bits: 0:63)";
  std::vector<bool> data_decwext = data->gtFdlWord().gtDecisionWordExtended();
  std::vector<bool> emul_decwext = emul->gtFdlWord().gtDecisionWordExtended();
  m_dumpFile << "\n\tdata: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (data_decwext.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << "\n\temul: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (emul_decwext.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << std::endl;

  m_dumpFile << "\n\tTechnical triggers (bits: 0:63)";
  std::vector<bool> data_fdlttw = data->gtFdlWord().gtTechnicalTriggerWord();
  std::vector<bool> emul_fdlttw = emul->gtFdlWord().gtTechnicalTriggerWord();
  assert((int)data_fdlttw.size()==nbitword); 
  m_dumpFile << "\n\tdata: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (data_fdlttw.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << "\n\temul: ";
  for(int i=0; i<nbitword; i++) {
    if (i%16==0) m_dumpFile << " ";
    m_dumpFile << (emul_fdlttw.at(nbitword-1-i)  ? '1' : '0');
  }
  m_dumpFile << std::endl;

  m_dumpFile << "\n\tL1GtFdlWord";
  m_dumpFile << "\n\tdata: "
	     << " BoardId:"   << data->gtFdlWord().boardId()
	     << " BxInEvent:" << data->gtFdlWord().bxInEvent()
	     << " BxNr:"      << data->gtFdlWord().bxNr()
	     << " EventNr:"   << data->gtFdlWord().eventNr()
	     << " NoAlgo:"    << data->gtFdlWord().noAlgo()
	     << " FinalOR:"   << data->gtFdlWord().finalOR()
	     << " LocalBxNr:" << data->gtFdlWord().localBxNr();
  m_dumpFile << "\n\temul: "
	     << " BoardId:"   << emul->gtFdlWord().boardId()
	     << " BxInEvent:" << emul->gtFdlWord().bxInEvent()
	     << " BxNr:"      << emul->gtFdlWord().bxNr()
	     << " EventNr:"   << emul->gtFdlWord().eventNr()
	     << " NoAlgo:"    << emul->gtFdlWord().noAlgo()
	     << " FinalOR:"   << emul->gtFdlWord().finalOR()
	     << " LocalBxNr:" << emul->gtFdlWord().localBxNr()
	     << std::endl;

  m_dumpFile << "\n\tL1GtfeWord";
  m_dumpFile << "\n\tdata: " 
	     << " BoardId:"        << data->gtfeWord().boardId()
	     << " RecordLength:"   << data->gtfeWord().recordLength()
	     << " BxNr:"           << data->gtfeWord().bxNr() 
	     << " SetupVersion:"   << data->gtfeWord().setupVersion()
	     << " ActiveBoards:"   << data->gtfeWord().activeBoards()
	     << " TotalTriggerNr:" << data->gtfeWord().totalTriggerNr();
  m_dumpFile << "\n\temul: " 
	     << " BoardId:"        << emul->gtfeWord().boardId()
	     << " RecordLength:"   << emul->gtfeWord().recordLength()
	     << " BxNr:"           << emul->gtfeWord().bxNr() 
	     << " SetupVersion:"   << emul->gtfeWord().setupVersion()
	     << " ActiveBoards:"   << emul->gtfeWord().activeBoards()
	     << " TotalTriggerNr:" << emul->gtfeWord().totalTriggerNr()
	     << std::endl;

  // -- tcs 
  m_dumpFile << "\n\ttcsWord";
  m_dumpFile << "\n\tdata:"
	     << " DaqNr:"              << data->tcsWord().daqNr()
	     << " TriggerType:"        << data->tcsWord().triggerType()
	     << " Status:"             << data->tcsWord().status()
	     << " BxNr:"               << data->tcsWord().bxNr()
	     << " PartTrigNr:"         << data->tcsWord().partTrigNr()
	     << " EventNr:"            << data->tcsWord().eventNr() << "\n\t"
	     << " AssignedPartitions:" << data->tcsWord().assignedPartitions()
	     << " PartRunNr:"          << data->tcsWord().partTrigNr()
	     << " OrbitNr:"            << data->tcsWord().orbitNr();
  m_dumpFile << "\n\temul:"	     	      				     
	     << " DaqNr:"              << emul->tcsWord().daqNr()
	     << " TriggerType:"        << emul->tcsWord().triggerType()
	     << " Status:"             << emul->tcsWord().status()
	     << " BxNr:"               << emul->tcsWord().bxNr()
	     << " PartTrigNr:"         << emul->tcsWord().partTrigNr()
	     << " EventNr:"            << emul->tcsWord().eventNr() << "\n\t"       
	     << " AssignedPartitions:" << emul->tcsWord().assignedPartitions()
	     << " PartRunNr:"          << emul->tcsWord().partTrigNr()
	     << " OrbitNr:"            << emul->tcsWord().orbitNr()
	     << "\n" << std::endl;

  char ok[10];
  if(match) sprintf(ok,"successful");
  else      sprintf(ok,"failed");
  m_dumpFile << " ...L1GlobalTriggerEvmReadoutRecord data and emulator comparison: " 
	     << ok << std::endl;

  return match;
}

/*following record is not produced by hardware, included for sw dump/tests only*/
bool
L1Comparator::compareCollections(edm::Handle<L1GlobalTriggerObjectMapRecord> data, 
				 edm::Handle<L1GlobalTriggerObjectMapRecord> emul) {

  m_dumpFile << "\n  L1GlobalTriggerObjectMapRecord candidates...\n";

  bool match = true;
  //match &= (*data==*emul);

  const std::vector<L1GlobalTriggerObjectMap>& data_ovec = data->gtObjectMap();
  const std::vector<L1GlobalTriggerObjectMap>& emul_ovec = emul->gtObjectMap();

  for(std::vector<L1GtLogicParser::OperandToken>::size_type idx=0; idx<data_ovec.size(); idx++) {
    match &= ( data_ovec.at(idx).algoName()               == emul_ovec.at(idx).algoName()               );
    match &= ( data_ovec.at(idx).algoBitNumber()          == emul_ovec.at(idx).algoBitNumber()	        );
    match &= ( data_ovec.at(idx).algoGtlResult()          == emul_ovec.at(idx).algoGtlResult()	        );
    match &= ( data_ovec.at(idx).combinationVector()      == emul_ovec.at(idx).combinationVector()	);
    match &= ( data_ovec.at(idx).operandTokenVector().size()==emul_ovec.at(idx).operandTokenVector().size());
    if(match) {
      for(std::vector<L1GtLogicParser::OperandToken>::size_type i=0; i<data_ovec.at(idx).operandTokenVector().size(); i++) {
	match &= ( data_ovec.at(idx).operandTokenVector().at(i).tokenName ==
		   emul_ovec.at(idx).operandTokenVector().at(i).tokenName );
	match &= ( data_ovec.at(idx).operandTokenVector().at(i).tokenNumber ==
		   emul_ovec.at(idx).operandTokenVector().at(i).tokenNumber );
	match &= ( data_ovec.at(idx).operandTokenVector().at(i).tokenResult ==
		   emul_ovec.at(idx).operandTokenVector().at(i).tokenResult );
      }
    }
  }

  if(m_dumpMode==0 && match)
    return match;
  
  // dump
  int idx = 0;
  m_dumpFile << "\n\tL1GlobalTriggerObjectMap";
  m_dumpFile << "\n\tdata: "
	     << " algorithmName:"         << data_ovec.at(idx).algoName()
	     << " Bitnumber:"             << data_ovec.at(idx).algoBitNumber()
	     << " GTLresult:"             << data_ovec.at(idx).algoGtlResult()
	     << " combinationVectorSize:" << data_ovec.at(idx).combinationVector().size()
	     << " operandTokenVector:"    << data_ovec.at(idx).operandTokenVector().size(); 
  m_dumpFile << "\n\temul: "
	     << " algorithmName:"         << emul_ovec.at(idx).algoName()
	     << " Bitnumber:"             << emul_ovec.at(idx).algoBitNumber()
	     << " GTLresult:"             << emul_ovec.at(idx).algoGtlResult()
	     << " combinationVectorSize:" << emul_ovec.at(idx).combinationVector().size()
	     << " operandTokenVector:"    << emul_ovec.at(idx).operandTokenVector().size() 
	     << "\n" << std::endl;

  char ok[10];
  if(match) sprintf(ok,"successful");
  else      sprintf(ok,"failed");
  m_dumpFile << " ...L1GlobalTriggerObjectMapRecord data and emulator comparison: " 
	     << ok << std::endl;
  
  return match;
}


bool
L1Comparator::compareFedRawCollections(edm::Handle<FEDRawDataCollection> data, 
				       edm::Handle<FEDRawDataCollection> emul, int fedId) {
  if(verbose())
    std::cout << "[L1Comparator]  fedraw start processing :" << std::endl << std::flush;
  if(dumpEvent_) {
    m_dumpFile << "\nEvent: " << nevt_ << std::endl;
    dumpEvent_=false;
  }
  m_dumpFile << "\n  FEDRawData candidates...\n";
  const FEDRawData& raw_fed_data = data->FEDData(fedId);
  const FEDRawData& raw_fed_emul = emul->FEDData(fedId);
  bool raw_match=true;
  for(int i=0; i!=(int)raw_fed_data.size();i++) {
    raw_match &= ( raw_fed_data.data()[i] == raw_fed_emul.data()[i] );
  }
  unsigned long dd = 0, de = 0;
  for(int i=0; i<(int)raw_fed_data.size()/4;i++) {
    dd=0; de=0;
    for(int j=0; j<4; j++)
      dd += ((raw_fed_data.data()[i*4+j]&0xff)<<(8*j));
    for(int j=0; j<4; j++) 
      de += ((raw_fed_emul.data()[i*4+j]&0xff)<<(8*j));
    if(m_dumpMode==-1 || (m_dumpMode==1 && dd!=de) ) {
      m_dumpFile << "\n\tdata: " << std::setw(8) << std::setfill('0') << std::hex << dd;
      m_dumpFile << "\n\temul: " << std::setw(8) << std::setfill('0') << std::hex << de;
    }
    m_dumpFile << std::endl;
  }
  char ok[10];
  if(raw_match) sprintf(ok,"successful");
  else          sprintf(ok,"failed");
  m_dumpFile << " ...FEDRawData data and emulator comparison: " 
	     << ok << std::endl;
  return raw_match;
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
    m_dumpFile << " #cand mismatch!"
	       << "\tdata: " << ndata
	       << "\temul: " << nemul
	       << std::endl;
  }
  col_it itd = data -> begin();
  col_it itm = emul -> begin();
  for (col_sz i=0; i<ndata; i++) {
    match &= dumpCandidate(*itd++,*itm++, m_dumpFile);
  }  
  return match; 
}

template <class T> 
bool L1Comparator::dumpCandidate( const T& dt, const T& em, std::ostream& s) {
  if(dt==em)
    return true;
  s<<dt<<std::endl; 
  s<<em<<std::endl<<std::endl;
  return false;
}
