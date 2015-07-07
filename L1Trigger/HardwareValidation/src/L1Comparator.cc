#include "L1Trigger/HardwareValidation/interface/L1Comparator.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace dedefs;

L1Comparator::L1Comparator(const edm::ParameterSet& iConfig) {

  verbose_ = iConfig.getUntrackedParameter<int>("VerboseFlag",0);
  m_stage1_layer2_ = iConfig.getParameter<bool>("stage1_layer2_");

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

  m_DEsource[RCT][0] = iConfig.getParameter<edm::InputTag>("RCTsourceData");
  m_DEsource[RCT][1] = iConfig.getParameter<edm::InputTag>("RCTsourceEmul");

  m_DEsource[GCT][0] = iConfig.getParameter<edm::InputTag>("GCTsourceData");
  m_DEsource[GCT][1] = iConfig.getParameter<edm::InputTag>("GCTsourceEmul");

  m_DEsource[DTP][0] = iConfig.getParameter<edm::InputTag>("DTPsourceData");
  m_DEsource[DTP][1] = iConfig.getParameter<edm::InputTag>("DTPsourceEmul");

  m_DEsource[DTF][0] = iConfig.getParameter<edm::InputTag>("DTFsourceData");
  m_DEsource[DTF][1] = iConfig.getParameter<edm::InputTag>("DTFsourceEmul");

  m_DEsource[RPC][0] = iConfig.getParameter<edm::InputTag>("RPCsourceData");
  m_DEsource[RPC][1] = iConfig.getParameter<edm::InputTag>("RPCsourceEmul");

  m_DEsource[GMT][0] = iConfig.getParameter<edm::InputTag>("GMTsourceData");
  m_DEsource[GMT][1] = iConfig.getParameter<edm::InputTag>("GMTsourceEmul");

  for(int sys=0; sys<DEnsys; sys++) {
    std::string data_label = SystLabel[sys] + "sourceData";
    std::string emul_label = SystLabel[sys] + "sourceEmul";

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

void L1Comparator::beginRun(edm::Run const& iRun, const edm::EventSetup& iSetup) {

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
  edm::Handle<L1GctJetCandCollection> gct_isotaujets_data;
  edm::Handle<L1GctJetCandCollection> gct_isotaujets_emul;
      
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
    if(m_stage1_layer2_ == false) {
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
    if(m_stage1_layer2_ == true){
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
      iEvent.getByLabel(m_DEsource[GCT][0].label(),"isoTauJets", gct_isotaujets_data);
      iEvent.getByLabel(m_DEsource[GCT][1].label(),"isoTauJets", gct_isotaujets_emul);
   
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
  edm::Handle<L1MuDTTrackContainer>       dtf_trk_data_;
  edm::Handle<L1MuDTTrackContainer>       dtf_trk_emul_;
  L1MuRegionalCandCollection  const* dtf_trk_data = 0;
  L1MuRegionalCandCollection  const* dtf_trk_emul = 0;
  if(m_doSys[DTF]) {
    iEvent.getByLabel(m_DEsource[DTF][0].label(),"DATA",dtf_trk_data_);
    iEvent.getByLabel(m_DEsource[DTF][1].label(),"DTTF",dtf_trk_emul_);
  } 
  //extract the regional cands
  typedef std::vector<L1MuDTTrackCand> L1MuDTTrackCandCollection;
  L1MuRegionalCandCollection dtf_trk_data_v, dtf_trk_emul_v;
  dtf_trk_data_v.clear(); dtf_trk_emul_v.clear();
  if(dtf_trk_data_.isValid()) {
    L1MuDTTrackCandCollection const *dttc = dtf_trk_data_->getContainer();
    for(L1MuDTTrackCandCollection::const_iterator  it=dttc->begin(); 
	it!=dttc->end(); it++)
      dtf_trk_data_v.push_back(L1MuRegionalCand(*it)); 
  }
  if(dtf_trk_emul_.isValid()) {
    L1MuDTTrackCandCollection const *dttc = dtf_trk_emul_->getContainer();
    for(L1MuDTTrackCandCollection::const_iterator  it=dttc->begin(); 
	it!=dttc->end(); it++)
      dtf_trk_emul_v.push_back(L1MuRegionalCand(*it)); 
  }  
  dtf_trk_data =&dtf_trk_data_v;
  dtf_trk_emul =&dtf_trk_emul_v;
  
  
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
  L1MuRegionalCandCollection  const* gmt_rdt_data(new L1MuRegionalCandCollection);
  L1MuRegionalCandCollection  const* gmt_rdt_emul(new L1MuRegionalCandCollection);
  //tbd: may compare extended candidates
  L1MuGMTCandCollection  const *gmt_can_data(new L1MuGMTCandCollection);
  L1MuGMTCandCollection  const *gmt_can_emul(new L1MuGMTCandCollection);
  if(m_doSys[GMT]) {
    iEvent.getByLabel(m_DEsource[GMT][0], gmt_data);
    iEvent.getByLabel(m_DEsource[GMT][1], gmt_emul);
    iEvent.getByLabel(m_DEsource[GMT][0], gmt_rdt_data_);
    iEvent.getByLabel(m_DEsource[GMT][1], gmt_rdt_emul_);
  }  
  L1MuGMTCandCollection       gmt_can_data_vec, gmt_can_emul_vec;
  L1MuRegionalCandCollection  gmt_rdt_data_vec, gmt_rdt_emul_vec;
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

  ///--- done getting collections. --- 

  //check collections validity
  bool isValidDE[DEnsys][2];
  for(int i=0; i<DEnsys; i++) for(int j=0; j<2; j++) isValidDE[i][j]=false;
  
  isValidDE[RCT][0] =      rct_em_data .isValid(); isValidDE[RCT][1] =    rct_em_emul .isValid();
  isValidDE[RCT][0]&=     rct_rgn_data .isValid(); isValidDE[RCT][1] =    rct_rgn_emul .isValid();

  if(m_stage1_layer2_ == false){
    isValidDE[GCT][0] = gct_isolaem_data .isValid(); isValidDE[GCT][1] = gct_isolaem_emul .isValid();
    isValidDE[GCT][0]&= gct_noisoem_data .isValid(); isValidDE[GCT][1]&= gct_noisoem_emul .isValid();
    isValidDE[GCT][0]&= gct_cenjets_data .isValid(); isValidDE[GCT][1]&= gct_cenjets_emul .isValid();
    isValidDE[GCT][0]&= gct_forjets_data .isValid(); isValidDE[GCT][1]&= gct_forjets_emul .isValid();
    isValidDE[GCT][0]&= gct_taujets_data .isValid(); isValidDE[GCT][1]&= gct_taujets_emul .isValid();
    isValidDE[GCT][0]&=  gct_etmiss_data .isValid(); isValidDE[GCT][1]&= gct_etmiss_emul .isValid();
    isValidDE[GCT][0]&=  gct_ettota_data .isValid(); isValidDE[GCT][1]&= gct_ettota_emul .isValid();
    isValidDE[GCT][0]&=  gct_htmiss_data .isValid(); isValidDE[GCT][1]&= gct_htmiss_emul .isValid();
    isValidDE[GCT][0]&=  gct_hfring_data .isValid(); isValidDE[GCT][1]&= gct_hfring_emul .isValid();
    isValidDE[GCT][0]&=  gct_hfbcnt_data .isValid(); isValidDE[GCT][1]&= gct_hfbcnt_emul .isValid();
  }

  if(m_stage1_layer2_ == true){
    isValidDE[GCT][0] = gct_isolaem_data .isValid();
    isValidDE[GCT][1] = gct_isolaem_emul .isValid();
    isValidDE[GCT][0]&= gct_noisoem_data .isValid();
    isValidDE[GCT][1]&= gct_noisoem_emul .isValid();
    isValidDE[GCT][0]&= gct_cenjets_data .isValid();
    isValidDE[GCT][1]&= gct_cenjets_emul .isValid();
    isValidDE[GCT][0]&= gct_forjets_data .isValid();
    isValidDE[GCT][1]&= gct_forjets_emul .isValid();
    isValidDE[GCT][0]&= gct_taujets_data .isValid();
    isValidDE[GCT][1]&= gct_taujets_emul .isValid();
    isValidDE[GCT][0]&= gct_isotaujets_data .isValid();
    isValidDE[GCT][1]&= gct_isotaujets_emul .isValid();
    isValidDE[GCT][0]&= gct_etmiss_data .isValid();
    isValidDE[GCT][1]&= gct_etmiss_emul .isValid();
    isValidDE[GCT][0]&= gct_ettota_data .isValid();
    isValidDE[GCT][1]&= gct_ettota_emul .isValid();
    isValidDE[GCT][0]&= gct_htmiss_data .isValid();
    isValidDE[GCT][1]&= gct_htmiss_emul .isValid();
    isValidDE[GCT][0]&= gct_hfring_data .isValid();
    isValidDE[GCT][1]&= gct_hfring_emul .isValid();
    isValidDE[GCT][0]&= gct_hfbcnt_data .isValid();
    isValidDE[GCT][1]&= gct_hfbcnt_emul .isValid();
  }
  
  isValidDE[DTP][0] =      dtp_ph_data_.isValid(); isValidDE[DTP][1] =     dtp_ph_emul_.isValid();
  isValidDE[DTP][0]&=      dtp_th_data_.isValid(); isValidDE[DTP][1]&=     dtp_th_emul_.isValid();

  isValidDE[DTF][0] =     dtf_trk_data_.isValid(); isValidDE[DTF][1] =    dtf_trk_emul_.isValid();

  isValidDE[RPC][0] =     rpc_cen_data .isValid(); isValidDE[RPC][1] =    rpc_cen_emul .isValid();
  isValidDE[RPC][0]&=     rpc_for_data .isValid(); isValidDE[RPC][1]&=    rpc_for_emul .isValid();

  isValidDE[LTC][0] =         ltc_data .isValid(); isValidDE[LTC][1] =        ltc_emul .isValid();

  isValidDE[GMT][0] =         gmt_data .isValid(); isValidDE[GMT][1] =        gmt_emul .isValid();

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
  if(m_doSys[RCT]&&isValid[RCT]) process<L1CaloEmCollection>            ( rct_em_data,      rct_em_emul, RCT,RCTem);
  if(m_doSys[RCT]&&isValid[RCT]) process<L1CaloRegionCollection>        ( rct_rgn_data,     rct_rgn_emul, RCT,RCTrgn);
  
  if(m_stage1_layer2_==false){
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEmCandCollection>       (gct_isolaem_data, gct_isolaem_emul, GCT,GCTisolaem);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEmCandCollection>       (gct_noisoem_data, gct_noisoem_emul, GCT,GCTnoisoem);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctJetCandCollection>      (gct_cenjets_data, gct_cenjets_emul, GCT,GCTcenjets);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctJetCandCollection>      (gct_forjets_data, gct_forjets_emul, GCT,GCTforjets);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctJetCandCollection>      (gct_taujets_data, gct_taujets_emul, GCT,GCTtaujets);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEtHadCollection>	(gct_ht_data,      gct_ht_emul,      GCT,GCTethad);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEtMissCollection>	(gct_etmiss_data,  gct_etmiss_emul,  GCT,GCTetmiss);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEtTotalCollection>	(gct_ettota_data , gct_ettota_emul,  GCT,GCTettot);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctHtMissCollection>	(gct_htmiss_data,  gct_htmiss_emul,  GCT,GCThtmiss);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctHFRingEtSumsCollection>	(gct_hfring_data,  gct_hfring_emul,  GCT,GCThfring);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctHFBitCountsCollection>	(gct_hfbcnt_data,  gct_hfbcnt_emul,  GCT,GCThfbit);
  }
  
  if(m_stage1_layer2_ == true){
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEmCandCollection>       (gct_isolaem_data,   gct_isolaem_emul,    GCT, GCTisolaem);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEmCandCollection>       (gct_noisoem_data,   gct_noisoem_emul,    GCT, GCTnoisoem);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctJetCandCollection>      (gct_cenjets_data,   gct_cenjets_emul,    GCT, GCTcenjets);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctJetCandCollection>      (gct_forjets_data,   gct_forjets_emul,    GCT, GCTforjets);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctJetCandCollection>      (gct_taujets_data,   gct_taujets_emul,    GCT, GCTtaujets);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctJetCandCollection>      (gct_isotaujets_data,gct_isotaujets_emul, GCT, GCTisotaujets);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEtHadCollection>	(gct_ht_data,        gct_ht_emul,         GCT, GCTethad);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEtMissCollection>	(gct_etmiss_data,    gct_etmiss_emul,     GCT, GCTetmiss);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctEtTotalCollection>	(gct_ettota_data ,   gct_ettota_emul,     GCT, GCTettot);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctHtMissCollection>	(gct_htmiss_data,    gct_htmiss_emul,     GCT, GCThtmiss);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctHFRingEtSumsCollection>	(gct_hfring_data,    gct_hfring_emul,     GCT, GCThfring);
    if(m_doSys[GCT]&&isValid[GCT]) process<L1GctHFBitCountsCollection>	(gct_hfbcnt_data,    gct_hfbcnt_emul,     GCT, GCThfbit);
  }

  if(m_doSys[DTP]&&isValid[DTP]) process<L1MuDTChambPhDigiCollection>   ( dtp_ph_data,      dtp_ph_emul, DTP,DTtpPh);
  if(m_doSys[DTP]&&isValid[DTP]) process<L1MuDTChambThDigiCollection>   ( dtp_th_data,      dtp_th_emul, DTP,DTtpTh);

  if(m_doSys[DTF]&&isValid[DTF]) process<L1MuRegionalCandCollection>    ( dtf_trk_data,     dtf_trk_emul, DTF,DTtftrk);

  if(m_doSys[RPC]&&isValid[RPC]) process<L1MuRegionalCandCollection>    ( rpc_cen_data,     rpc_cen_emul, RPC,RPCcen);
  if(m_doSys[RPC]&&isValid[RPC]) process<L1MuRegionalCandCollection>    ( rpc_for_data,     rpc_for_emul, RPC,RPCfor);

  if(m_doSys[GMT]&&isValid[GMT]) process<L1MuGMTCandCollection>         ( gmt_data,         gmt_emul,     GMT,GMTmain);
  if(m_doSys[GMT]&&isValid[GMT]) process<L1MuRegionalCandCollection>    ( gmt_rdt_data,     gmt_rdt_emul, GMT,GMTrdt);
  if(m_doSys[GMT]&&isValid[GMT]) process<L1MuGMTCandCollection>         ( gmt_can_data,     gmt_can_emul, GMT,GMTcnd);

  // >>---- GLT ---- <<  
  GltDEDigi gltdigimon;
  

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
  
  if(verbose())
    std::cout << "L1comparator::analize() end. " << nevt_ << std::endl;
  
}


template <class T> 
void L1Comparator::process(T const* data, T const* emul, const int sys, const int cid) {

  if(verbose())
    std::cout << "L1Comparator::process -ing system:" << sys 
	      << " (" << SystLabel[sys] << "), data type " << cid 
	      << "...\n" << std::endl;
  if(verbose())
    std::cout << "L1Comparator::process debug "
	      << " (size "  << data->size() << "," <<emul->size() << ")"   
	      << ".\n" << std::endl;

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
	      << ".\n" << std::endl;
  
  if(ndata==0&&nemul==0) {
    if(verbose())
      std::cout << "L1Comparator::process " 
	        << "empty collections -- exiting!\n" << std::endl;
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

template <class myCol> 
bool L1Comparator::CompareCollections(edm::Handle<myCol> data, edm::Handle<myCol> emul) {
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
