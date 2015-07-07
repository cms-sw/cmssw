#include "DQM/L1TMonitor/interface/L1TdeGCT.h"
#include <bitset>

using namespace dedefs;

const int L1TdeGCT::nGctColl_;
const int L1TdeGCT::nerr;

L1TdeGCT::L1TdeGCT(const edm::ParameterSet& iConfig) {

  verbose_ = iConfig.getUntrackedParameter<int>("VerboseFlag",0);

  if(verbose())
    std::cout << "L1TdeGCT::L1TdeGCT()...\n" << std::flush;
  
  DEsource_ = consumes<L1DataEmulRecord>(iConfig.getParameter<edm::InputTag>("DataEmulCompareSource"));
  histFolder_ = iConfig.getUntrackedParameter<std::string>("HistFolder");
  
  histFile_ = iConfig.getUntrackedParameter<std::string>("HistFile", "");
  if(iConfig.getUntrackedParameter<bool> ("disableROOToutput", true))
    histFile_ = "";

  if (histFile_.size()!=0) {
    edm::LogInfo("OutputRootFile") 
      << "L1TEmulator GCT specific histograms will be saved to " 
      << histFile_.c_str() 
      << std::endl;
  }

  hasRecord_=true;
  
  if(verbose())
    std::cout << "L1TdeGCT::L1TdeGCT()...done.\n" << std::flush;
  m_stage1_layer2_ = iConfig.getParameter<bool>("stage1_layer2_");
  
}

L1TdeGCT::~L1TdeGCT() {}

void L1TdeGCT::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& evSetup)
{}

void L1TdeGCT::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& evSetup)
{}

void L1TdeGCT::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&){

  int    rnkNBins = 63;
  double rnkMinim = 0.5;
  double rnkMaxim = 63.5;

  
  if(verbose())
    std::cout << "L1TdeGCT::beginRun()  start\n" << std::flush;

  ibooker.setCurrentFolder(histFolder_);
  //for Legacy GCT
  if (m_stage1_layer2_ == false) {
  
    sysrates = ibooker.book1D("sysrates","RATE OF COMPARISON FAILURES",nGctColl_, 0, nGctColl_ );

    for(int j=0; j<2; j++) {
      std::string lbl("sysncand"); 
      lbl += (j==0?"Data":"Emul");
      std::string title("GCT OBJECT MULTIPLICITY ");
      title += (j==0?"(DATA)":"(EMULATOR)");
      sysncand[j] = ibooker.book1D(lbl.data(),title.data(),nGctColl_, 0, nGctColl_ );
    }
    
    for(int j=0; j<nGctColl_; j++) {
      
      ibooker.setCurrentFolder(std::string(histFolder_+"/"+cLabel[j]));
      
      std::string lbl("");
      std::string title("");
      lbl.clear();
      title.clear();
      lbl+=cLabel[j];lbl+="ErrorFlag"; 
      title+=cLabel[j];title+=" ErrorFlag"; 
      errortype[j] = ibooker.book1D(lbl.data(),title.data(), nerr, 0, nerr);
      
      lbl.clear();
      title.clear();
      lbl+=cLabel[j];lbl+="Eta"; 
      title+=cLabel[j];title+=" ETA OF COMPARISON FAILURES"; 
      eta[j] = ibooker.book1D(lbl.data(),title.data(),
			   etaNBins, etaMinim, etaMaxim);
      lbl.clear();
      title.clear();
      lbl+=cLabel[j];lbl+="Phi"; 
      title+=cLabel[j];title+=" PHI OF COMPARISON FAILURES"; 
      phi[j] = ibooker.book1D(lbl.data(),title.data(),
			   phiNBins, phiMinim, phiMaxim);

      lbl.clear();
      title.clear();
      lbl+=cLabel[j];lbl+="Etaphi"; 
      title+=cLabel[j];title+=" ETA PHI OF COMPARISON FAILURES"; 
      etaphi[j] = ibooker.book2D(lbl.data(),title.data(), 
			      etaNBins, etaMinim, etaMaxim,
			      phiNBins, phiMinim, phiMaxim
			      );
      lbl.clear();
      title.clear();
      lbl+=cLabel[j];lbl+="Rank";
      title+=cLabel[j];title+=" RANK OF COMPARISON FAILURES"; 
      rnk[j] = ibooker.book1D(lbl.data(),title.data(),
			       rnkNBins, rnkMinim, rnkMaxim);
      //
      lbl.clear();
      title.clear();
      lbl+=cLabel[j];lbl+="Eta"; lbl+="Data";
      title+=cLabel[j];title+=" ETA (DATA)"; 
      etaData[j] = ibooker.book1D(lbl.data(),title.data(),
			       etaNBins, etaMinim, etaMaxim);
      lbl.clear();
      title.clear();
      lbl+=cLabel[j];lbl+="Phi";  lbl+="Data";
      title+=cLabel[j];title+=" PHI (DATA)"; 
      phiData[j] = ibooker.book1D(lbl.data(),title.data(),
			       phiNBins, phiMinim, phiMaxim);

      lbl.clear();
      title.clear();
      lbl+=cLabel[j];lbl+="Rank";  lbl+="Data";
      title+=cLabel[j];title+=" RANK (DATA)"; 
      rnkData[j] = ibooker.book1D(lbl.data(),title.data(),
			       rnkNBins, rnkMinim, rnkMaxim);
      lbl.clear();
      lbl+=cLabel[j];lbl+="Dword"; 
      dword[j] = ibooker.book1D(lbl.data(),lbl.data(),nbit,0,nbit);
      lbl.clear();
      lbl+=cLabel[j];lbl+="Eword"; 
      eword[j] = ibooker.book1D(lbl.data(),lbl.data(),nbit,0,nbit);
      lbl.clear();
      lbl+=cLabel[j];lbl+="DEword"; 
      deword[j] = ibooker.book1D(lbl.data(),lbl.data(),nbit,0,nbit);
      //lbl.clear();
      //lbl+=cLabel[j];lbl+="Masked"; 
      //masked[j] = dbe->book1D(lbl.data(),lbl.data(),nbit,0,nbit);
    }
    
    for(int i=0; i<nGctColl_; i++) {
      sysrates   ->setBinLabel(i+1,cLabel[i]);
      sysncand[0]->setBinLabel(i+1,cLabel[i]);
      sysncand[1]->setBinLabel(i+1,cLabel[i]);
    }

    for(int i=0; i<nGctColl_; i++) {
      for(int j=0; j<nerr; j++) {
        errortype[i]->setBinLabel(j+1,errLabel[j]);
      }
    }
  
    for(int i=0; i<nGctColl_; i++) {
      etaphi [i]->setAxisTitle("GCT #eta",1);
      etaphi [i]->setAxisTitle("GCT #phi",2);
      eta    [i]->setAxisTitle("GCT #eta");
      phi    [i]->setAxisTitle("GCT #phi");
      rnk    [i]->setAxisTitle("Rank");
      etaData[i]->setAxisTitle("GCT #eta");
      phiData[i]->setAxisTitle("GCT #phi");
      rnkData[i]->setAxisTitle("Rank");
      dword  [i]->setAxisTitle("trigger data word bit");
      eword  [i]->setAxisTitle("trigger data word bit");
      deword [i]->setAxisTitle("trigger data word bit");
    }

    for(int i=0; i<nGctColl_; i++) {
      colCount[i]=0;
      nWithCol[i]=0;
    }
  }

  // for stage 1 layer 2 
  
  if (m_stage1_layer2_ == true) {

    sysrates = ibooker.book1D("sysrates","RATE OF COMPARISON FAILURES",nStage1Layer2Coll_, 0, nStage1Layer2Coll_ );

    for(int j=0; j<2; j++) {
      std::string lbl("sysncand"); 
      lbl += (j==0?"Data":"Emul");
      std::string title("Stage1Layer2 OBJECT MULTIPLICITY ");
      title += (j==0?"(DATA)":"(EMULATOR)");
      sysncand[j] = ibooker.book1D(lbl.data(),title.data(),nStage1Layer2Coll_, 0, nStage1Layer2Coll_ );
    }
        
    for(int j=0; j<nStage1Layer2Coll_; j++) {
      
      ibooker.setCurrentFolder(std::string(histFolder_+"/"+sLabel[j]));
      
      if(sLabel[j]=="MHT"){
	rnkNBins = 127;
	rnkMinim = 0.5;
	rnkMaxim = 127.5;
      }

      if((sLabel[j]=="HT")||(sLabel[j]=="ET")||(sLabel[j]=="MET")){
	rnkNBins = 4096;
	rnkMinim = -0.5;
	rnkMaxim = 4095.5;
      }

      if(sLabel[j]=="Stage1HFSums"){
	rnkNBins = 9;
	rnkMinim = -0.5;
	rnkMaxim = 8.5;
      }

      if(sLabel[j]=="IsoTauJet"){
	rnkNBins = 64;
	rnkMinim = -0.5;
	rnkMaxim = 63.5;
      }
      
      std::string lbl("");
      std::string title("");
      lbl.clear();
      title.clear();
      lbl+=sLabel[j];lbl+="ErrorFlag"; 
      title+=sLabel[j];title+=" ErrorFlag"; 
      errortype_stage1layer2[j] = ibooker.book1D(lbl.data(),title.data(), nerr, 0, nerr);
      
      lbl.clear();
      title.clear();
      lbl+=sLabel[j];lbl+="Eta"; 
      title+=sLabel[j];title+=" ETA OF COMPARISON FAILURES"; 
      eta_stage1layer2[j] = ibooker.book1D(lbl.data(),title.data(),
			   etaNBins, etaMinim, etaMaxim);
      lbl.clear();
      title.clear();
      lbl+=sLabel[j];lbl+="Phi"; 
      title+=sLabel[j];title+=" PHI OF COMPARISON FAILURES"; 
      phi_stage1layer2[j] = ibooker.book1D(lbl.data(),title.data(),
			   phiNBins, phiMinim, phiMaxim);

      lbl.clear();
      title.clear();
      lbl+=sLabel[j];lbl+="Etaphi"; 
      title+=sLabel[j];title+=" ETA PHI OF COMPARISON FAILURES"; 
      etaphi_stage1layer2[j] = ibooker.book2D(lbl.data(),title.data(), 
			      etaNBins, etaMinim, etaMaxim,
			      phiNBins, phiMinim, phiMaxim);
      lbl.clear();
      title.clear();
      lbl+=sLabel[j];lbl+="Rank";
      title+=sLabel[j];title+=" RANK OF COMPARISON FAILURES"; 
      rnk_stage1layer2[j] = ibooker.book1D(lbl.data(),title.data(),
			       rnkNBins, rnkMinim, rnkMaxim);
      //
      lbl.clear();
      title.clear();
      lbl+=sLabel[j];lbl+="Eta"; lbl+="Data";
      title+=sLabel[j];title+=" ETA (DATA)"; 
      etaData_stage1layer2[j] = ibooker.book1D(lbl.data(),title.data(),
			       etaNBins, etaMinim, etaMaxim);
      lbl.clear();
      title.clear();
      lbl+=sLabel[j];lbl+="Phi";  lbl+="Data";
      title+=sLabel[j];title+=" PHI (DATA)"; 
      phiData_stage1layer2[j] = ibooker.book1D(lbl.data(),title.data(),
			       phiNBins, phiMinim, phiMaxim);

      lbl.clear();
      title.clear();
      lbl+=sLabel[j];lbl+="Rank";  lbl+="Data";
      title+=sLabel[j];title+=" RANK (DATA)"; 
      rnkData_stage1layer2[j] = ibooker.book1D(lbl.data(),title.data(),
			       rnkNBins, rnkMinim, rnkMaxim);
      lbl.clear();
      lbl+=sLabel[j];lbl+="Dword"; 
      dword_stage1layer2[j] = ibooker.book1D(lbl.data(),lbl.data(),nbit,0,nbit);
      lbl.clear();
      lbl+=sLabel[j];lbl+="Eword"; 
      eword_stage1layer2[j] = ibooker.book1D(lbl.data(),lbl.data(),nbit,0,nbit);
      lbl.clear();
      lbl+=sLabel[j];lbl+="DEword"; 
      deword_stage1layer2[j] = ibooker.book1D(lbl.data(),lbl.data(),nbit,0,nbit);
      //lbl.clear();
      //lbl+=cLabel[j];lbl+="Masked"; 
      //masked_stage1layer2[j] = dbe->book1D(lbl.data(),lbl.data(),nbit,0,nbit);
    }
    
    for(int i=0; i<nStage1Layer2Coll_; i++) {
      sysrates   ->setBinLabel(i+1,sLabel[i]);
      sysncand[0]->setBinLabel(i+1,sLabel[i]);
      sysncand[1]->setBinLabel(i+1,sLabel[i]);
    }

    for(int i=0; i<nStage1Layer2Coll_; i++) {
      for(int j=0; j<nerr; j++) {
        errortype_stage1layer2[i]->setBinLabel(j+1,errLabel[j]);
      }
    }
  
    for(int i=0; i<nStage1Layer2Coll_; i++) {
      etaphi_stage1layer2[i]->setAxisTitle("Stage1Layer2 #eta",1);
      etaphi_stage1layer2[i]->setAxisTitle("Stage1Layer2 #phi",2);
      eta_stage1layer2[i]->setAxisTitle("Stage1Layer2 #eta");
      phi_stage1layer2[i]->setAxisTitle("Stage1Layer2 #phi");
      rnk_stage1layer2[i]->setAxisTitle("Rank");
      etaData_stage1layer2[i]->setAxisTitle("Stage1Layer2 #eta");
      phiData_stage1layer2[i]->setAxisTitle("Stage1Layer2 #phi");
      rnkData_stage1layer2[i]->setAxisTitle("Rank");
      dword_stage1layer2  [i]->setAxisTitle("trigger data word bit");
      eword_stage1layer2  [i]->setAxisTitle("trigger data word bit");
      deword_stage1layer2 [i]->setAxisTitle("trigger data word bit");
    }

    for(int i=0; i<nStage1Layer2Coll_; i++) {
      colCount_stage1Layer2[i]=0;
      nWithCol_stage1Layer2[i]=0;
    }
  }
  
  if(verbose())
    std::cout << "L1TdeGCT::beginJob()  end.\n" << std::flush;
}

// ------------ method called to for each event  ------------


void L1TdeGCT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  if(!hasRecord_)
    return;
  
  if(verbose())
    std::cout << "L1TdeGCT::analyze()  start\n" << std::flush;

  /// get the comparison results
  edm::Handle<L1DataEmulRecord> deRecord;
  iEvent.getByToken(DEsource_, deRecord);

  if (!deRecord.isValid()) {
    edm::LogInfo("DataNotFound") 
      << "Cannot find L1DataEmulRecord"
      << " Please verify that comparator was successfully executed."
      << " Emulator DQM for GCT will be skipped!"
      << std::endl;
    hasRecord_=false;
    return;
  }

  if (m_stage1_layer2_ == false) {
    bool isComp = deRecord->get_isComp(GCT);
    if(!isComp) {
      if(verbose()) 
        std::cout << "[L1TdeGCT] Gct information not generated in de-record."
		<< " Skiping event!\n" << std::flush;
      return;
    }

    int DEncand[2]={0};
    for(int j=0; j<2; j++) 
      DEncand[j] = deRecord->getNCand(GCT,j);
  
    if(verbose()) 
      std::cout << "[L1TdeGCT] ncands" 
	        << " data: " << DEncand[0]
	        << " emul: " << DEncand[1]
	        << std::endl;

    /// get the de candidates
    L1DEDigiCollection deColl;
    deColl = deRecord->getColl();

    // extract the GCT comparison digis
    L1DEDigiCollection gctColl;

    gctColl.reserve(20);
    gctColl.clear();


    for(L1DEDigiCollection::const_iterator it=deColl.begin(); 
        it!=deColl.end(); it++) 
      if(!it->empty()) 
        if(it->sid()==GCT)
	  gctColl.push_back(*it);
  
    if(verbose()) {
      std::cout << "[L1TdeGCT] record has " << gctColl.size() 
    	        << " gct de digis\n" << std::flush;
      for(L1DEDigiCollection::const_iterator it=gctColl.begin(); 
	  it!=gctColl.end(); it++)
        std::cout << "\t" << *it << std::endl;
    }

    const int nullVal = L1DataEmulDigi().reset();

    /// --- Fill histograms(me) ---
  
    // d|e candidate loop
    for(L1DEDigiCollection::const_iterator it=gctColl.begin(); 
        it!=gctColl.end(); it++) {
    
      // sid should be GCT
      int sid = it->sid();
      // cid: GCTisolaem, GCTnoisoem, GCTcenjets, GCTforjets, GCTtaujets
      int cid = it->cid();
      ///(note see L1Trigger/HardwareValidation/interface/DEtrait.h)

      if(verbose()) 
        std::cout << "[L1TdeGCT] processing digi "
	    	  << " sys:"  << sid
		  << " type:" << cid
		  << " \n\t"
		  << *it << "\n" 
		  << std::flush;
    
      //assert(cid==GCT);
      if(sid!=GCT || it->empty()) {
        LogDebug("L1TdeGCT") << "consistency check failure, non-gct digis!";
        continue;
      }
    
      int type    = it->type();
      double phiv = it->x1();
      double etav = it->x2();
      float rankarr[2]; 
      it->rank(rankarr);
      float rnkv = rankarr[0];

      double wei = 1.;

      unsigned int mask = (~0x0);

      // shift coll type for starting at zero
      int ccid = cid - dedefs::GCTisolaem; 
      if(ccid<0 || ccid >= nGctColl_) {
        LogDebug("L1TdeGCT") << "consistency check failure, col type outbounds:"
	        	     << ccid << "\n";
        ccid=0;
      }
    
      //type: 0:agree 1:loc.agree, 2:loc.disagree, 3:data.only, 4:emul.only
      if(it->type()<4) 
        sysncand[0]->Fill(ccid); 
      if(it->type()<5&&it->type()!=3) 
        sysncand[1]->Fill(ccid);
    
      errortype[ccid]->Fill(type);

      wei=1.; if(!type) wei=0.;
      if(etav!=nullVal && phiv!=nullVal)
        etaphi[ccid]->Fill(etav,phiv,wei);
      if(etav!=nullVal)
        eta   [ccid]->Fill(etav,wei);
      if(phiv!=nullVal)
        phi   [ccid]->Fill(phiv,wei);
      rnk[ccid]->Fill(rnkv,wei);    

      //exclude e-only cands (only data)
      wei=1.;if(type==4) wei=0.;
      if(etav!=nullVal)
        etaData[ccid]->Fill(etav,wei);
      if(phiv!=nullVal)
        phiData[ccid]->Fill(phiv,wei);
      rnkData[ccid]->Fill(rnkv,wei);
      wei=1;

      // GCT trigger bits
      unsigned int word[2];
      it->data(word);
      std::bitset<32> dbits(word[0]);
      std::bitset<32> ebits(word[1]);
      unsigned int dexor = ( (word[0]) ^ (word[1]) );
      //disagreeing bits
      std::bitset<32> debits(dexor);
      //disagreeing bits after masking
      std::bitset<32> dembits( ( (dexor) & (mask) ) );
    
      if(verbose())
        std::cout << "l1degct" 
		  << " sid:" << sid << " cid:" << cid << "\n"
		  << " data:0x" << std::hex << word[0] << std::dec
		  << " bitset:" << dbits
		  << "\n"
		  << " emul:0x" << std::hex << word[1] << std::dec
		  << " bitset:" << ebits
		  << "\n"
		  << "  xor:0x" << std::hex << dexor << std::dec
		  << " bitset:" << debits
		  << " bitset:" << ( (dbits) ^ (ebits) )
		  << "\n" << std::flush;

      ///bitset loop
      for(int ibit=0; ibit<32; ibit++) {
        wei=1.;
        //comparison gives no info if there's only 1 candidate
        if(type==3 || type==4) wei=0.; 
        if(dbits  [ibit]) dword[ccid]->Fill(ibit,wei);
        if(ebits  [ibit]) eword[ccid]->Fill(ibit,wei);
        if(debits [ibit])deword[ccid]->Fill(ibit,wei);
        //if(dembits[ibit])masked[sid]->Fill(ibit,wei);
      }
      wei=1;
    
    }

    //error rates per GCT trigger object type
    int hasCol[nGctColl_]={0};
    int nagree[nGctColl_]={0};
    for(L1DEDigiCollection::const_iterator it=gctColl.begin(); 
        it!=gctColl.end(); it++) {
      int ccid = it->cid()-dedefs::GCTisolaem;
      ccid = (ccid<0 || ccid >= nGctColl_) ? 0:ccid;
      hasCol[ccid]++;
      if(!it->type()) 
        nagree[ccid]++;
    }
    for(int i=0; i<nGctColl_; i++) {
      if(!hasCol[i]) continue;
    ///event based rate
    //nWithCol[i]++;
    //if(nagree[i]<hasCol[i]) colCount[i]++;
    ///object based rate
      nWithCol[i]+=hasCol[i];//#of objects
      colCount[i]+=nagree[i];//#of agreements
    }
    for(int i=0; i<nGctColl_; i++) {
      int ibin = i+1;
      double rate = nWithCol[i] ? 1.-1.*colCount[i]/nWithCol[i]: 0.;
      sysrates->setBinContent(ibin,rate);
      if(verbose()) {
        std::cout << "[L1TDEMON] analyze rate computation\t\n"
		<< " colid:"   << i
		<< "(so far)"
		<< " nWithCol: " << nWithCol[i]
		<< " colCount: " << colCount[i]
		<< "(this event)"
		<< "hasCol: "    << hasCol[i] 
		<< " nagree: "   << nagree[i]
		<< " rate:"    << sysrates->getBinContent(ibin) 
		<< "\n" << std::flush;
        if(rate>1. || rate<0.)
	  std::cout << "problem, error rate for " << SystLabel[i] 
		  <<" is "<<sysrates->getBinContent(ibin)
		  << "\n" << std::flush;
      }
    }
  }

  if (m_stage1_layer2_ == true) {
    bool isComp = deRecord->get_isComp(GCT);
    if(!isComp) {
      if(verbose()) 
        std::cout << "[L1TdeGCT] Gct information not generated in de-record."
		<< " Skiping event!\n" << std::flush;
      return;
    }

    int DEncand[2]={0};
    for(int j=0; j<2; j++) 
      DEncand[j] = deRecord->getNCand(GCT,j);
  
    if(verbose()) 
      std::cout << "[L1TdeGCT] ncands" 
	        << " data: " << DEncand[0]
	        << " emul: " << DEncand[1]
	        << std::endl;
    
    /// get the de candidates
    L1DEDigiCollection deColl;
    deColl = deRecord->getColl();

    // extract the GCT comparison digis
    L1DEDigiCollection stage1layer2Coll;

    stage1layer2Coll.reserve(21);
    stage1layer2Coll.clear();


    for(L1DEDigiCollection::const_iterator it=deColl.begin(); 
        it!=deColl.end(); it++) 
      if(!it->empty()) 
        if(it->sid()==GCT)
	  stage1layer2Coll.push_back(*it);
  
    if(verbose()) {
      std::cout << "[L1TdeSTAGE1LAYER2] record has " << stage1layer2Coll.size() 
    	        << " stage1layer2 de digis\n" << std::endl;
      for(L1DEDigiCollection::const_iterator it=stage1layer2Coll.begin(); 
	  it!=stage1layer2Coll.end(); it++)
        std::cout << "\t" << *it << std::endl;
    }

    const int nullVal = L1DataEmulDigi().reset();

    /// --- Fill histograms(me) ---
  
    // d|e candidate loop
    for(L1DEDigiCollection::const_iterator it=stage1layer2Coll.begin(); 
        it!=stage1layer2Coll.end(); it++) {
    
      // sid should be GCT
      int sid = it->sid();

      int cid = it->cid();
      ///(note see L1Trigger/HardwareValidation/interface/DEtrait.h)

      if(verbose()) 
        std::cout << "[L1TdeStage1Layer2] processing digi "
	    	  << " sys:"  << sid
		  << " type:" << cid
		  << " \n\t"
		  << *it << "\n" 
		  << std::endl;
    
      if(sid!=GCT || it->empty()) {
        LogDebug("L1TdeGCT") << "consistency check failure, non-stage1layer2 digis!";
        continue;
      }
    
      int type    = it->type();
      double phiv = it->x1();
      double etav = it->x2();
      float rankarr[2]; 
      it->rank(rankarr);
      float rnkv = rankarr[0];

      double wei = 1.;

      unsigned int mask = (~0x0);

      // shift coll type for starting at zero
      int ccid = cid - dedefs::GCTisolaem; 
      if(ccid<0 || ccid >= nStage1Layer2Coll_) {
        LogDebug("L1TdeGCT") << "consistency check failure, col type outbounds:"
	        	     << ccid << "\n";
        ccid=0;
      }
    
      //type: 0:agree 1:loc.agree, 2:loc.disagree, 3:data.only, 4:emul.only
      if(it->type()<4) 
        sysncand[0]->Fill(ccid); 
      if(it->type()<5&&it->type()!=3) 
        sysncand[1]->Fill(ccid);
      errortype_stage1layer2[ccid]->Fill(type);
      wei=1.; if(!type) wei=0.;
      if(etav!=nullVal && phiv!=nullVal)
        etaphi_stage1layer2[ccid]->Fill(etav,phiv,wei);
      if(etav!=nullVal)
        eta_stage1layer2   [ccid]->Fill(etav,wei);
      if(phiv!=nullVal)
        phi_stage1layer2   [ccid]->Fill(phiv,wei);
      rnk_stage1layer2[ccid]->Fill(rnkv,wei);    

      //exclude e-only cands (only data)
      wei=1.;if(type==4) wei=0.;
      if(etav!=nullVal)
        etaData_stage1layer2[ccid]->Fill(etav,wei);
      if(phiv!=nullVal)
        phiData_stage1layer2[ccid]->Fill(phiv,wei);
      rnkData_stage1layer2[ccid]->Fill(rnkv,wei);
      wei=1;
      
      // GCT trigger bits
      unsigned int word_stage1layer2[2];
      it->data(word_stage1layer2);
      std::bitset<32> dbits(word_stage1layer2[0]);
      std::bitset<32> ebits(word_stage1layer2[1]);
      unsigned int dexor = ( (word_stage1layer2[0]) ^ (word_stage1layer2[1]) );
      //disagreeing bits
      std::bitset<32> debits(dexor);
      //disagreeing bits after masking
      std::bitset<32> dembits( ( (dexor) & (mask) ) );
         
      if(verbose())
        std::cout << "l1degct" 
  	    	  << " sid:" << sid << " cid:" << cid << "\n"
		  << " data:0x" << std::hex << word_stage1layer2[0] << std::dec
		  << " bitset:" << dbits
		  << "\n"
		  << " emul:0x" << std::hex << word_stage1layer2[1] << std::dec
		  << " bitset:" << ebits
		  << "\n"
		  << "  xor:0x" << std::hex << dexor << std::dec
		  << " bitset:" << debits
		  << " bitset:" << ( (dbits) ^ (ebits) )
		  << "\n" << std::flush;

      ///bitset loop
      for(int ibit=0; ibit<32; ibit++) {
        wei=1.;
        //comparison gives no info if there's only 1 candidate
        if(type==3 || type==4) wei=0.; 
        if(dbits  [ibit]) dword_stage1layer2[ccid]->Fill(ibit,wei);	
        if(ebits  [ibit]) eword_stage1layer2[ccid]->Fill(ibit,wei);
        if(debits [ibit])deword_stage1layer2[ccid]->Fill(ibit,wei);
        //if(dembits[ibit])masked[sid]->Fill(ibit,wei);
      }
      wei=1;
    
    }
    //error rates per GCT trigger object type
    int hasCol[nStage1Layer2Coll_]={0};
    int nagree[nStage1Layer2Coll_]={0};
    for(L1DEDigiCollection::const_iterator it=stage1layer2Coll.begin(); 
        it!=stage1layer2Coll.end(); it++) {
      int ccid = it->cid()-dedefs::GCTisolaem;
      ccid = (ccid<0 || ccid >= nStage1Layer2Coll_) ? 0:ccid;
      hasCol[ccid]++;
      if(!it->type()) 
        nagree[ccid]++;
    }
    for(int i=0; i<nStage1Layer2Coll_; i++) {
      if(!hasCol[i]) continue;
    ///event based rate
    //nWithCol[i]++;
    //if(nagree[i]<hasCol[i]) colCount[i]++;
    ///object based rate
      nWithCol_stage1Layer2[i]+=hasCol[i];//#of objects
      colCount_stage1Layer2[i]+=nagree[i];//#of agreements
    }
    for(int i=0; i<nStage1Layer2Coll_; i++) {
      int ibin = i+1;
      double rate = nWithCol_stage1Layer2[i] ? 1.-1.*colCount_stage1Layer2[i]/nWithCol_stage1Layer2[i]: 0.;
      sysrates->setBinContent(ibin,rate);
      if(verbose()) {
        std::cout << "[L1TDEMON] analyze rate computation\t\n"
		  << " colid:"   << i
		  << "(so far)"
		  << " nWithCol: " << nWithCol_stage1Layer2[i]
		  << " colCount: " << colCount_stage1Layer2[i]
		  << "(this event)"
		  << "hasCol: "    << hasCol[i] 
		  << " nagree: "   << nagree[i]
		  << " rate:"    << sysrates->getBinContent(ibin) 
		  << "\n" << std::flush;
        if(rate>1. || rate<0.)
	  std::cout << "problem, error rate for " << SystLabel[i] 
		  <<" is "<<sysrates->getBinContent(ibin)
		  << "\n" << std::flush;
      }
    }
  }
}


