#include "DQM/L1TMonitor/interface/L1TdeGCT.h"
#include <bitset>

using namespace dedefs;

L1TdeGCT::L1TdeGCT(const edm::ParameterSet& iConfig) {

  verbose_ = iConfig.getUntrackedParameter<int>("VerboseFlag",0);

  if(verbose())
    std::cout << "L1TdeGCT::L1TdeGCT()...\n" << std::flush;
  
  DEsource_ = iConfig.getParameter<edm::InputTag>("DataEmulCompareSource");
  histFolder_ = iConfig.getUntrackedParameter<std::string>("HistFolder", "L1TEMU/GCTexpert/");
  
  dbe = NULL;
  if (iConfig.getUntrackedParameter<bool>("DQMStore", false)) { 
    dbe = edm::Service<DQMStore>().operator->();
    dbe->setVerbose(0);
  }
  
  histFile_ = iConfig.getUntrackedParameter<std::string>("HistFile", "");
  if(iConfig.getUntrackedParameter<bool> ("disableROOToutput", true))
    histFile_ = "";

  if (histFile_.size()!=0) {
    edm::LogInfo("OutputRootFile") 
      << "L1TEmulator GCT specific histograms will be saved to " 
      << histFile_.c_str() 
      << std::endl;
  }

  if(dbe!=NULL)
    dbe->setCurrentFolder(histFolder_);

  hasRecord_=true;
  
  if(verbose())
    std::cout << "L1TdeGCT::L1TdeGCT()...done.\n" << std::flush;
}

L1TdeGCT::~L1TdeGCT() {}

void 
L1TdeGCT::beginJob(const edm::EventSetup&) {

  if(verbose())
    std::cout << "L1TdeGCT::beginJob()  start\n" << std::flush;

  DQMStore* dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
  if(dbe) {
    dbe->setCurrentFolder(histFolder_);
    dbe->rmdir(histFolder_);
  }

  // (em) iso, no-iso, (jets) cen, for, tau
  std::string cLabel[nGctColl_]= 
    {"IsoEM", "NoisoEM", "CenJet", "ForJet", "TauJet"};
  const int nerr  = 5; 
  const int nbit = 32;
  
  if(dbe) {
    dbe->setCurrentFolder(histFolder_);

    // book histograms here 

    const int    phiNBins = 18  ;
    const double phiMinim = -0.5;
    const double phiMaxim = 17.5;
    const int    etaNBins = 22  ;
    const double etaMinim = -0.5;
    const double etaMaxim = 21.5;
    const int    rnkNBins = 63;
    const double rnkMinim = 0.5;
    const double rnkMaxim = 63.5;

    sysrates = dbe->book1D("sysrates","sysrates",nGctColl_, 0, nGctColl_ );

    for(int j=0; j<2; j++) {
      std::string lbl("sysncand"); 
      lbl += (j==0?"Data":"Emul");
      sysncand[j] = dbe->book1D(lbl.data(),lbl.data(),nGctColl_, 0, nGctColl_ );
    }
    
    for(int j=0; j<nGctColl_; j++) {
      
      dbe->setCurrentFolder(std::string(histFolder_+cLabel[j]));
      
      std::string lbl("");
      lbl.clear();
      lbl+=cLabel[j];lbl+="ErrorFlag"; 
      errortype[j] = dbe->book1D(lbl.data(),lbl.data(), nerr, 0, nerr);
      
      lbl.clear();
      lbl+=cLabel[j];lbl+="Eta"; 
      eta[j] = dbe->book1D(lbl.data(),lbl.data(),
			   etaNBins, etaMinim, etaMaxim);
      lbl.clear();
      lbl+=cLabel[j];lbl+="Phi"; 
      phi[j] = dbe->book1D(lbl.data(),lbl.data(),
			   phiNBins, phiMinim, phiMaxim);

      lbl.clear();
      lbl+=cLabel[j];lbl+="Etaphi"; 
      etaphi[j] = dbe->book2D(lbl.data(),lbl.data(), 
			      etaNBins, etaMinim, etaMaxim,
			      phiNBins, phiMinim, phiMaxim
			      );
      //
      lbl.clear();
      lbl+=cLabel[j];lbl+="Eta"; lbl+="Data";
      etaData[j] = dbe->book1D(lbl.data(),lbl.data(),
			       etaNBins, etaMinim, etaMaxim);
      lbl.clear();
      lbl+=cLabel[j];lbl+="Phi";  lbl+="Data";
      phiData[j] = dbe->book1D(lbl.data(),lbl.data(),
			       phiNBins, phiMinim, phiMaxim);

      lbl.clear();
      lbl+=cLabel[j];lbl+="Rank";  lbl+="Data";
      rnkData[j] = dbe->book1D(lbl.data(),lbl.data(),
			       rnkNBins, rnkMinim, rnkMaxim);
      lbl.clear();
      lbl+=cLabel[j];lbl+="Dword"; 
      dword[j] = dbe->book1D(lbl.data(),lbl.data(),nbit,0,nbit);
      lbl.clear();
      lbl+=cLabel[j];lbl+="Eword"; 
      eword[j] = dbe->book1D(lbl.data(),lbl.data(),nbit,0,nbit);
      lbl.clear();
      lbl+=cLabel[j];lbl+="DEword"; 
      deword[j] = dbe->book1D(lbl.data(),lbl.data(),nbit,0,nbit);
      lbl.clear();
      lbl+=cLabel[j];lbl+="Masked"; 
      masked[j] = dbe->book1D(lbl.data(),lbl.data(),nbit,0,nbit);
    }

  }
  
  /// labeling
  std::string errLabel[nerr]= {
    "Agree", "Loc. Agree", "L.Disagree", "Data only", "Emul only"
  };
  
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
    etaphi [i]->setAxisTitle("eta",1);
    etaphi [i]->setAxisTitle("phi",2);
    eta    [i]->setAxisTitle("eta");
    phi    [i]->setAxisTitle("phi");
    etaData[i]->setAxisTitle("eta");
    phiData[i]->setAxisTitle("phi");
    rnkData[i]->setAxisTitle("rank");
    dword  [i]->setAxisTitle("trigger data word bit");
    eword  [i]->setAxisTitle("trigger data word bit");
    deword [i]->setAxisTitle("trigger data word bit");
    masked [i]->setAxisTitle("trigger data word bit");
  }

  for(int i=0; i<nGctColl_; i++) {
    colCount[i]=0;
    nWithCol[i]=0;
  }
  
  if(verbose())
    std::cout << "L1TdeGCT::beginJob()  end.\n" << std::flush;
}

void 
L1TdeGCT::endJob() {
  if(verbose())
    std::cout << "L1TdeGCT::endJob()...\n" << std::flush;

  if(histFile_.size()!=0  && dbe) 
    dbe->save(histFile_);

  if(verbose())
    std::cout << "L1TdeGCT::endJob()  end.\n" << std::flush;
}

 
// ------------ method called to for each event  ------------
void
  L1TdeGCT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  if(!hasRecord_)
    return;
  
  if(verbose())
    std::cout << "L1TdeGCT::analyze()  start\n" << std::flush;

  /// get the comparison results
  edm::Handle<L1DataEmulRecord> deRecord;
  iEvent.getByLabel(DEsource_, deRecord);

  if (!deRecord.isValid()) {
    edm::LogInfo("DataNotFound") 
      << "Cannot find L1DataEmulRecord with label "
      << DEsource_.label() 
      << " Please verify that comparator was successfully executed."
      << " Emulator DQM for GCT will be skipped!"
      << std::endl;
    hasRecord_=false;
    return;
  }

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
      etaphi[sid]->Fill(etav,phiv,wei);
    if(etav!=nullVal)
      eta   [sid]->Fill(etav,wei);
    if(phiv!=nullVal)
      phi   [sid]->Fill(phiv,wei);
    

    //exclude e-only cands (only data)
    wei=1.;if(type==4) wei=0.;
    if(etav!=nullVal)
      etaData[sid]->Fill(etav,wei);
    if(phiv!=nullVal)
      phiData[sid]->Fill(phiv,wei);
    rnkData[sid]->Fill(rnkv,wei);
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
      if(dbits  [ibit]) dword[sid]->Fill(ibit,wei);
      if(ebits  [ibit]) eword[sid]->Fill(ibit,wei);
      if(debits [ibit])deword[sid]->Fill(ibit,wei);
      if(dembits[ibit])masked[sid]->Fill(ibit,wei);
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
    nWithCol[i]++;
    if(nagree[i]<hasCol[i])
      colCount[i]++;
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

    
  if(verbose())
    std::cout << "L1TdeGCT::analyze() end.\n" << std::flush;
  
}

