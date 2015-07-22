#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>
#include <stdlib.h>
#include <string.h>

#include "HLTrigger/HLTanalyzers/interface/HLTInfo.h"
#include "FWCore/Common/interface/TriggerNames.h"

// L1 related
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

HLTInfo::HLTInfo() {

  //set parameter defaults 
  _Debug=false;
  _OR_BXes=false;
  UnpackBxInEvent=1;
}

void HLTInfo::beginRun(const edm::Run& run, const edm::EventSetup& c){ 


  bool changed(true);
  if (hltPrescaleProvider_->init(run,c,processName_,changed)) {
    // if init returns TRUE, initialisation has succeeded!
    if (changed) {
      // The HLT config has actually changed wrt the previous Run, hence rebook your
      // histograms or do anything else dependent on the revised HLT config
      std::cout << "Initalizing HLTConfigProvider"  << std::endl;
    }
  } else {
    // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
    // with the file and/or code and needs to be investigated!
    std::cout << " HLT config extraction failure with process name " << processName_ << std::endl;
    // In this case, all access methods will return empty values!
  }

}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTInfo::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  processName_ = pSet.getParameter<std::string>("HLTProcessName") ;

  edm::ParameterSet myHltParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  std::vector<std::string> parameterNames = myHltParams.getParameterNames() ;
  
  for ( std::vector<std::string>::iterator iParam = parameterNames.begin();
        iParam != parameterNames.end(); iParam++ ){
    if ( (*iParam) == "Debug" ) _Debug =  myHltParams.getParameter<bool>( *iParam );
  }

  dummyBranches_ = pSet.getUntrackedParameter<std::vector<std::string> >("dummyBranches",std::vector<std::string>(0));

  HltEvtCnt = 0;
  const int kMaxTrigFlag = 10000;
  trigflag = new int[kMaxTrigFlag];
  trigPrescl = new int[kMaxTrigFlag];
  L1EvtCnt = 0;
  const int kMaxL1Flag = 10000;
  l1flag = new int[kMaxL1Flag];
  l1flag5Bx = new int[kMaxTrigFlag];
  l1Prescl = new int[kMaxL1Flag];
  l1techflag = new int[kMaxL1Flag];
  l1techflag5Bx = new int[kMaxTrigFlag];
  l1techPrescl = new int[kMaxTrigFlag];
  const int kMaxHLTPart = 10000;
  hltppt = new float[kMaxHLTPart];
  hltpeta = new float[kMaxHLTPart];
  const int kMaxL1ExtEmI = 10000;
  l1extiemet = new float[kMaxL1ExtEmI];
  l1extieme = new float[kMaxL1ExtEmI];
  l1extiemeta = new float[kMaxL1ExtEmI];
  l1extiemphi = new float[kMaxL1ExtEmI];
  const int kMaxL1ExtEmN = 10000;
  l1extnemet = new float[kMaxL1ExtEmN];
  l1extneme = new float[kMaxL1ExtEmN];
  l1extnemeta = new float[kMaxL1ExtEmN];
  l1extnemphi = new float[kMaxL1ExtEmN];
  const int kMaxL1ExtMu = 10000;
  l1extmupt = new float[kMaxL1ExtMu];
  l1extmue = new float[kMaxL1ExtMu];
  l1extmueta = new float[kMaxL1ExtMu];
  l1extmuphi = new float[kMaxL1ExtMu];
  l1extmuiso = new int[kMaxL1ExtMu];
  l1extmumip = new int[kMaxL1ExtMu];
  l1extmufor = new int[kMaxL1ExtMu];
  l1extmurpc = new int[kMaxL1ExtMu];
  l1extmuqul = new int[kMaxL1ExtMu];
  l1extmuchg = new int[kMaxL1ExtMu];
  const int kMaxL1ExtJtC = 10000;
  l1extjtcet = new float[kMaxL1ExtJtC];
  l1extjtce = new float[kMaxL1ExtJtC];
  l1extjtceta = new float[kMaxL1ExtJtC];
  l1extjtcphi = new float[kMaxL1ExtJtC];
  const int kMaxL1ExtJtF = 10000;
  l1extjtfet = new float[kMaxL1ExtJtF];
  l1extjtfe = new float[kMaxL1ExtJtF];
  l1extjtfeta = new float[kMaxL1ExtJtF];
  l1extjtfphi = new float[kMaxL1ExtJtF];
  const int kMaxL1ExtJt = 10000;
  l1extjtet = new float[kMaxL1ExtJt];
  l1extjte = new float[kMaxL1ExtJt];
  l1extjteta = new float[kMaxL1ExtJt];
  l1extjtphi = new float[kMaxL1ExtJt];
  const int kMaxL1ExtTau = 10000;
  l1exttauet = new float[kMaxL1ExtTau];
  l1exttaue = new float[kMaxL1ExtTau];
  l1exttaueta = new float[kMaxL1ExtTau];
  l1exttauphi = new float[kMaxL1ExtTau];

  algoBitToName = new TString[128];
  techBitToName = new TString[128];

  HltTree->Branch("NL1IsolEm",&nl1extiem,"NL1IsolEm/I");
  HltTree->Branch("L1IsolEmEt",l1extiemet,"L1IsolEmEt[NL1IsolEm]/F");
  HltTree->Branch("L1IsolEmE",l1extieme,"L1IsolEmE[NL1IsolEm]/F");
  HltTree->Branch("L1IsolEmEta",l1extiemeta,"L1IsolEmEta[NL1IsolEm]/F");
  HltTree->Branch("L1IsolEmPhi",l1extiemphi,"L1IsolEmPhi[NL1IsolEm]/F");
  HltTree->Branch("NL1NIsolEm",&nl1extnem,"NL1NIsolEm/I");
  HltTree->Branch("L1NIsolEmEt",l1extnemet,"L1NIsolEmEt[NL1NIsolEm]/F");
  HltTree->Branch("L1NIsolEmE",l1extneme,"L1NIsolEmE[NL1NIsolEm]/F");
  HltTree->Branch("L1NIsolEmEta",l1extnemeta,"L1NIsolEmEta[NL1NIsolEm]/F");
  HltTree->Branch("L1NIsolEmPhi",l1extnemphi,"L1NIsolEmPhi[NL1NIsolEm]/F");
  HltTree->Branch("NL1Mu",&nl1extmu,"NL1Mu/I");
  HltTree->Branch("L1MuPt",l1extmupt,"L1MuPt[NL1Mu]/F");
  HltTree->Branch("L1MuE",l1extmue,"L1MuE[NL1Mu]/F");
  HltTree->Branch("L1MuEta",l1extmueta,"L1MuEta[NL1Mu]/F");
  HltTree->Branch("L1MuPhi",l1extmuphi,"L1MuPhi[NL1Mu]/F");
  HltTree->Branch("L1MuIsol",l1extmuiso,"L1MuIsol[NL1Mu]/I");
  HltTree->Branch("L1MuMip",l1extmumip,"L1MuMip[NL1Mu]/I");
  HltTree->Branch("L1MuFor",l1extmufor,"L1MuFor[NL1Mu]/I");
  HltTree->Branch("L1MuRPC",l1extmurpc,"L1MuRPC[NL1Mu]/I");
  HltTree->Branch("L1MuQal",l1extmuqul,"L1MuQal[NL1Mu]/I");
  HltTree->Branch("L1MuChg",l1extmuchg,"L1MuChg[NL1Mu]/I");
  HltTree->Branch("NL1CenJet",&nl1extjetc,"NL1CenJet/I");
  HltTree->Branch("L1CenJetEt",l1extjtcet,"L1CenJetEt[NL1CenJet]/F");
  HltTree->Branch("L1CenJetE",l1extjtce,"L1CenJetE[NL1CenJet]/F");
  HltTree->Branch("L1CenJetEta",l1extjtceta,"L1CenJetEta[NL1CenJet]/F");
  HltTree->Branch("L1CenJetPhi",l1extjtcphi,"L1CenJetPhi[NL1CenJet]/F");
  HltTree->Branch("NL1ForJet",&nl1extjetf,"NL1ForJet/I");
  HltTree->Branch("L1ForJetEt",l1extjtfet,"L1ForJetEt[NL1ForJet]/F");
  HltTree->Branch("L1ForJetE",l1extjtfe,"L1ForJetE[NL1ForJet]/F");
  HltTree->Branch("L1ForJetEta",l1extjtfeta,"L1ForJetEta[NL1ForJet]/F");
  HltTree->Branch("L1ForJetPhi",l1extjtfphi,"L1ForJetPhi[NL1ForJet]/F");
  /*
  HltTree->Branch("NL1Jet",&nl1extjet,"NL1Jet/I");
  HltTree->Branch("L1JetEt",l1extjtet,"L1JetEt[NL1Jet]/F");
  HltTree->Branch("L1JetE",l1extjte,"L1JetE[NL1Jet]/F");
  HltTree->Branch("L1JetEta",l1extjteta,"L1JetEta[NL1Jet]/F");
  HltTree->Branch("L1JetPhi",l1extjtphi,"L1JetPhi[NL1Jet]/F");
  */
  HltTree->Branch("NL1Tau",&nl1exttau,"NL1Tau/I");
  HltTree->Branch("L1TauEt",l1exttauet,"L1TauEt[NL1Tau]/F");
  HltTree->Branch("L1TauE",l1exttaue,"L1TauE[NL1Tau]/F");
  HltTree->Branch("L1TauEta",l1exttaueta,"L1TauEta[NL1Tau]/F");
  HltTree->Branch("L1TauPhi",l1exttauphi,"L1TauPhi[NL1Tau]/F");
  HltTree->Branch("L1Met",&met,"L1Met/F");
  HltTree->Branch("L1MetPhi",&metphi,"L1MetPhi/F");
  HltTree->Branch("L1EtTot",&ettot,"L1EtTot/F");
  HltTree->Branch("L1Mht",&mht,"L1Mht/F");
  HltTree->Branch("L1MhtPhi",&mhtphi,"L1MhtPhi/F");
  HltTree->Branch("L1EtHad",&ethad,"L1EtHad/F");

  // L1GctJetCounts
  HltTree->Branch("L1HfRing1EtSumPositiveEta",&l1hfRing1EtSumPositiveEta,"L1HfRing1EtSumPositiveEta/I");
  HltTree->Branch("L1HfRing2EtSumPositiveEta",&l1hfRing2EtSumPositiveEta,"L1HfRing2EtSumPositiveEta/I");
  HltTree->Branch("L1HfRing1EtSumNegativeEta",&l1hfRing1EtSumNegativeEta,"L1HfRing1EtSumNegativeEta/I");
  HltTree->Branch("L1HfRing2EtSumNegativeEta",&l1hfRing2EtSumNegativeEta,"L1HfRing2EtSumNegativeEta/I");
  HltTree->Branch("L1HfTowerCountPositiveEtaRing1",&l1hfTowerCountPositiveEtaRing1,"L1HfTowerCountPositiveEtaRing1/I");
  HltTree->Branch("L1HfTowerCountNegativeEtaRing1",&l1hfTowerCountNegativeEtaRing1,"L1HfTowerCountNegativeEtaRing1/I");
  HltTree->Branch("L1HfTowerCountPositiveEtaRing2",&l1hfTowerCountPositiveEtaRing2,"L1HfTowerCountPositiveEtaRing2/I");
  HltTree->Branch("L1HfTowerCountNegativeEtaRing2",&l1hfTowerCountNegativeEtaRing2,"L1HfTowerCountNegativeEtaRing2/I");
}

/* **Analyze the event** */
void HLTInfo::analyze(const edm::Handle<edm::TriggerResults>                 & hltresults,
                      const edm::Handle<l1extra::L1EmParticleCollection>     & L1ExtEmIsol,
                      const edm::Handle<l1extra::L1EmParticleCollection>     & L1ExtEmNIsol,
                      const edm::Handle<l1extra::L1MuonParticleCollection>   & L1ExtMu,
                      const edm::Handle<l1extra::L1JetParticleCollection>    & L1ExtJetC,
                      const edm::Handle<l1extra::L1JetParticleCollection>    & L1ExtJetF,
		      const edm::Handle<l1extra::L1JetParticleCollection>    & L1ExtJet,
                      const edm::Handle<l1extra::L1JetParticleCollection>    & L1ExtTau,
                      const edm::Handle<l1extra::L1EtMissParticleCollection> & L1ExtMet,
                      const edm::Handle<l1extra::L1EtMissParticleCollection> & L1ExtMht,
                      const edm::Handle<L1GlobalTriggerReadoutRecord>        & L1GTRR,
		      const edm::Handle<L1GctHFBitCountsCollection>          & gctBitCounts,
		      const edm::Handle<L1GctHFRingEtSumsCollection>         & gctRingSums,
		      edm::EventSetup const& eventSetup,
		      edm::Event const& iEvent,
                      TTree* HltTree) {

//   std::cout << " Beginning HLTInfo " << std::endl;


  /////////// Analyzing HLT Trigger Results (TriggerResults) //////////
  if (hltresults.isValid()) {
    int ntrigs = hltresults->size();
    if (ntrigs==0){std::cout << "%HLTInfo -- No trigger name given in TriggerResults of the input " << std::endl;}

    edm::TriggerNames const& triggerNames = iEvent.triggerNames(*hltresults);

    // 1st event : Book as many branches as trigger paths provided in the input...
    if (HltEvtCnt==0){
      for (int itrig = 0; itrig != ntrigs; ++itrig) {
        TString trigName = triggerNames.triggerName(itrig);
        HltTree->Branch(trigName,trigflag+itrig,trigName+"/I");
        HltTree->Branch(trigName+"_Prescl",trigPrescl+itrig,trigName+"_Prescl/I");
      }

      int itdum = ntrigs;
      for (unsigned int idum = 0; idum < dummyBranches_.size(); ++idum) {
	TString trigName(dummyBranches_[idum].data());
	bool addThisBranch = 1;
	for (int itrig = 0; itrig != ntrigs; ++itrig) {
	  TString realTrigName = triggerNames.triggerName(itrig);
	  if(trigName == realTrigName) addThisBranch = 0;
	}
	if(addThisBranch){
	  HltTree->Branch(trigName,trigflag+itdum,trigName+"/I");
	  HltTree->Branch(trigName+"_Prescl",trigPrescl+itdum,trigName+"_Prescl/I");
	  trigflag[itdum] = 0;
	  trigPrescl[itdum] = 0;
	  ++itdum;
	}
      }

      HltEvtCnt++;
    }
    // ...Fill the corresponding accepts in branch-variables
    //HLTConfigProvider const&  hltConfig = hltPrescaleProvider_->hltConfigProvider();
    //std::cout << "Number of prescale sets: " << hltConfig.prescaleSize() << std::endl;
    //std::cout << "Number of HLT paths: " << hltConfig.size() << std::endl;
    //int presclSet = hltPrescaleProvider_->prescaleSet(iEvent, eventSetup);
    //std::cout<<"\tPrescale set number: "<< presclSet <<std::endl; 

    for (int itrig = 0; itrig != ntrigs; ++itrig){

      std::string trigName=triggerNames.triggerName(itrig);
      bool accept = hltresults->accept(itrig);

      trigPrescl[itrig] = hltPrescaleProvider_->prescaleValue(iEvent, eventSetup, trigName);


      if (accept){trigflag[itrig] = 1;}
      else {trigflag[itrig] = 0;}

      if (_Debug){
        if (_Debug) std::cout << "%HLTInfo --  Number of HLT Triggers: " << ntrigs << std::endl;
        std::cout << "%HLTInfo --  HLTTrigger(" << itrig << "): " << trigName << " = " << accept << std::endl;
      }
    }
  }
  else { if (_Debug) std::cout << "%HLTInfo -- No Trigger Result" << std::endl;}

  /////////// Analyzing L1Extra objects //////////

  const int maxL1EmIsol = 4;
  for (int i=0; i!=maxL1EmIsol; ++i){
    l1extiemet[i] = -999.;
    l1extieme[i] = -999.;
    l1extiemeta[i] = -999.;
    l1extiemphi[i] = -999.;
  }
  if (L1ExtEmIsol.isValid()) {
    nl1extiem = maxL1EmIsol;
    l1extra::L1EmParticleCollection myl1iems;
    myl1iems = * L1ExtEmIsol;
    std::sort(myl1iems.begin(),myl1iems.end(),EtGreater());
    int il1exem = 0;
    for (l1extra::L1EmParticleCollection::const_iterator emItr = myl1iems.begin(); emItr != myl1iems.end(); ++emItr) {
      l1extiemet[il1exem] = emItr->et();
      l1extieme[il1exem] = emItr->energy();
      l1extiemeta[il1exem] = emItr->eta();
      l1extiemphi[il1exem] = emItr->phi();
      il1exem++;
    }
  }
  else {
    nl1extiem = 0;
    if (_Debug) std::cout << "%HLTInfo -- No Isolated L1 EM object" << std::endl;
  }

  const int maxL1EmNIsol = 4;
  for (int i=0; i!=maxL1EmNIsol; ++i){
    l1extnemet[i] = -999.;
    l1extneme[i] = -999.;
    l1extnemeta[i] = -999.;
    l1extnemphi[i] = -999.;
  }
  if (L1ExtEmNIsol.isValid()) {
    nl1extnem = maxL1EmNIsol;
    l1extra::L1EmParticleCollection myl1nems;
    myl1nems = * L1ExtEmNIsol;
    std::sort(myl1nems.begin(),myl1nems.end(),EtGreater());
    int il1exem = 0;
    for (l1extra::L1EmParticleCollection::const_iterator emItr = myl1nems.begin(); emItr != myl1nems.end(); ++emItr) {
      l1extnemet[il1exem] = emItr->et();
      l1extneme[il1exem] = emItr->energy();
      l1extnemeta[il1exem] = emItr->eta();
      l1extnemphi[il1exem] = emItr->phi();
      il1exem++;
    }
  }
  else {
    nl1extnem = 0;
    if (_Debug) std::cout << "%HLTInfo -- No Non-Isolated L1 EM object" << std::endl;
  }

  const int maxL1Mu = 4;
  for (int i=0; i!=maxL1Mu; ++i){
    l1extmupt[i] = -999.;
    l1extmue[i] = -999.;
    l1extmueta[i] = -999.;
    l1extmuphi[i] = -999.;
    l1extmuiso[i] = -999;
    l1extmumip[i] = -999;
    l1extmufor[i] = -999;
    l1extmurpc[i] = -999;
    l1extmuqul[i] = -999;
    l1extmuchg[i] = -999;
  }
  if (L1ExtMu.isValid()) {
    nl1extmu = maxL1Mu;
    l1extra::L1MuonParticleCollection myl1mus;
    myl1mus = * L1ExtMu;
    std::sort(myl1mus.begin(),myl1mus.end(),PtGreater());
    int il1exmu = 0;
    for (l1extra::L1MuonParticleCollection::const_iterator muItr = myl1mus.begin(); muItr != myl1mus.end(); ++muItr) {
      l1extmupt[il1exmu] = muItr->pt();
      l1extmue[il1exmu] = muItr->energy();
      l1extmueta[il1exmu] = muItr->eta();
      l1extmuphi[il1exmu] = muItr->phi();
      l1extmuiso[il1exmu] = muItr->isIsolated(); // = 1 for Isolated ?
      l1extmumip[il1exmu] = muItr->isMip(); // = 1 for Mip ?
      l1extmufor[il1exmu] = muItr->isForward();
      l1extmurpc[il1exmu] = muItr->isRPC();
      l1extmuchg[il1exmu] = muItr->charge();
      L1MuGMTExtendedCand gmtCand = muItr->gmtMuonCand();
      l1extmuqul[il1exmu] = gmtCand.quality(); // Muon quality as defined in the GT
      il1exmu++;
    }
  }
  else {
    nl1extmu = 0;
    if (_Debug) std::cout << "%HLTInfo -- No L1 MU object" << std::endl;
  }

  const int maxL1CenJet = 4;
  for (int i=0; i!=maxL1CenJet; ++i){
    l1extjtcet[i] = -999.;
    l1extjtce[i] = -999.;
    l1extjtceta[i] = -999.;
    l1extjtcphi[i] = -999.;
  }
  if (L1ExtJetC.isValid()) {
    nl1extjetc = maxL1CenJet;
    l1extra::L1JetParticleCollection myl1jetsc;
    myl1jetsc = * L1ExtJetC;
    std::sort(myl1jetsc.begin(),myl1jetsc.end(),EtGreater());
    int il1exjt = 0;
    for (l1extra::L1JetParticleCollection::const_iterator jtItr = myl1jetsc.begin(); jtItr != myl1jetsc.end(); ++jtItr) {
      l1extjtcet[il1exjt] = jtItr->et();
      l1extjtce[il1exjt] = jtItr->energy();
      l1extjtceta[il1exjt] = jtItr->eta();
      l1extjtcphi[il1exjt] = jtItr->phi();
      il1exjt++;
    }
  }
  else {
    nl1extjetc = 0;
    if (_Debug) std::cout << "%HLTInfo -- No L1 Central JET object" << std::endl;
  }

  const int maxL1ForJet = 4;
  for (int i=0; i!=maxL1ForJet; ++i){
    l1extjtfet[i] = -999.;
    l1extjtfe[i] = -999.;
    l1extjtfeta[i] = -999.;
    l1extjtfphi[i] = -999.;
  }
  if (L1ExtJetF.isValid()) {
    nl1extjetf = maxL1ForJet;
    l1extra::L1JetParticleCollection myl1jetsf;
    myl1jetsf = * L1ExtJetF;
    std::sort(myl1jetsf.begin(),myl1jetsf.end(),EtGreater());
    int il1exjt = 0;
    for (l1extra::L1JetParticleCollection::const_iterator jtItr = myl1jetsf.begin(); jtItr != myl1jetsf.end(); ++jtItr) {
      l1extjtfet[il1exjt] = jtItr->et();
      l1extjtfe[il1exjt] = jtItr->energy();
      l1extjtfeta[il1exjt] = jtItr->eta();
      l1extjtfphi[il1exjt] = jtItr->phi();
      il1exjt++;
    }
  }
  else {
    nl1extjetf = 0;
    if (_Debug) std::cout << "%HLTInfo -- No L1 Forward JET object" << std::endl;
  }

  const int maxL1Jet = 324;
  for (int i=0; i!=maxL1Jet; ++i){
    l1extjtet[i] = -999.;
    l1extjte[i] = -999.;
    l1extjteta[i] = -999.;
    l1extjtphi[i] = -999.;
  }
  if (L1ExtJet.isValid()) {
    if (_Debug) std::cout << "%HLTInfo -- Found L1 JET object" << std::endl;
    nl1extjet = maxL1Jet;
    l1extra::L1JetParticleCollection myl1jets;
    myl1jets = * L1ExtJet;
    std::sort(myl1jets.begin(),myl1jets.end(),EtGreater());
    int il1exjt = 0;
    for (l1extra::L1JetParticleCollection::const_iterator jtItr = myl1jets.begin(); jtItr != myl1jets.end(); ++jtItr) {
      l1extjtet[il1exjt] = jtItr->et();
      l1extjte[il1exjt] = jtItr->energy();
      l1extjteta[il1exjt] = jtItr->eta();
      l1extjtphi[il1exjt] = jtItr->phi();
      il1exjt++;
    }
  }
  else {
    //    nl1extjetf = 0;
    if (_Debug) std::cout << "%HLTInfo -- No L1 JET object" << std::endl;
  }


  const int maxL1TauJet = 4;
  for (int i=0; i!=maxL1TauJet; ++i){
    l1exttauet[i] = -999.;
    l1exttaue[i] = -999.;
    l1exttaueta[i] = -999.;
    l1exttauphi[i] = -999.;
  }
  if (L1ExtTau.isValid()) {
    nl1exttau = maxL1TauJet;
    l1extra::L1JetParticleCollection myl1taus;
    myl1taus = * L1ExtTau;
    std::sort(myl1taus.begin(),myl1taus.end(),EtGreater());
    int il1extau = 0;
    for (l1extra::L1JetParticleCollection::const_iterator tauItr = myl1taus.begin(); tauItr != myl1taus.end(); ++tauItr) {
      l1exttauet[il1extau] = tauItr->et();
      l1exttaue[il1extau] = tauItr->energy();
      l1exttaueta[il1extau] = tauItr->eta();
      l1exttauphi[il1extau] = tauItr->phi();
      il1extau++;
    }
  }
  else {
    nl1exttau = 0;
    if (_Debug) std::cout << "%HLTInfo -- No L1 TAU object" << std::endl;
  }

  if (L1ExtMet.isValid()) {
    met    = L1ExtMet->begin()->etMiss();
    metphi = L1ExtMet->begin()->phi();
    ettot  = L1ExtMet->begin()->etTotal();
  }
  else {
    if (_Debug) std::cout << "%HLTInfo -- No L1 MET object" << std::endl;
  }

  if (L1ExtMht.isValid()) {
    mht    = L1ExtMht->begin()->etMiss();
    mhtphi = L1ExtMht->begin()->phi();
    ethad  = L1ExtMht->begin()->etTotal();
  }
  else {
    if (_Debug) std::cout << "%HLTInfo -- No L1 MHT object" << std::endl;
  }

  //==============L1 information=======================================

  // L1 Triggers from Menu
  L1GtUtils const& l1GtUtils = hltPrescaleProvider_->l1GtUtils();

  edm::ESHandle<L1GtTriggerMenu> menuRcd;
  eventSetup.get<L1GtTriggerMenuRcd>().get(menuRcd) ;
  const L1GtTriggerMenu* menu = menuRcd.product();

  int iErrorCode = -1;
  L1GtUtils::TriggerCategory trigCategory = L1GtUtils::AlgorithmTrigger;
  const int pfSetIndexAlgorithmTrigger = l1GtUtils.prescaleFactorSetIndex(
             iEvent, trigCategory, iErrorCode);
  if (iErrorCode == 0) {
    if (_Debug) std::cout << "%Prescale set index: " << pfSetIndexAlgorithmTrigger  << std::endl;
  }else{
    std::cout << "%Could not extract Prescale set index from event record. Error code: " << iErrorCode << std::endl;
  }

  // 1st event : Book as many branches as trigger paths provided in the input...
  if (L1GTRR.isValid()) {  

    DecisionWord gtDecisionWord = L1GTRR->decisionWord();
    const unsigned int numberTriggerBits(gtDecisionWord.size());
    const TechnicalTriggerWord&  technicalTriggerWordBeforeMask = L1GTRR->technicalTriggerWord();
    const unsigned int numberTechnicalTriggerBits(technicalTriggerWordBeforeMask.size());

    // 1st event : Book as many branches as trigger paths provided in the input...
    if (L1EvtCnt==0){

 
      //ccla determine if more than 1 bx was unpacked in event; add OR all bx's if so
      const edm::Provenance& prov = iEvent.getProvenance(L1GTRR.id());
      //const string& procName = prov.processName();
      //std::cout << "procName:" << procName << std::endl;
      //std::cout << "provinfo:" << prov << std::endl;
      //std::cout << "setid:" << setId << std::endl;
      edm::ParameterSet pSet=parameterSet(prov);
      //std::cout << "pset:" << pSet << std::endl;
      if (pSet.exists("UnpackBxInEvent")){
	UnpackBxInEvent = pSet.getParameter<int>("UnpackBxInEvent");
      }
      if (_Debug) std::cout << "Number of beam crossings unpacked by GT: " << UnpackBxInEvent << std::endl;
      if (UnpackBxInEvent == 5) _OR_BXes = true;

      // get L1 menu from event setup
      for (CItAlgo algo = menu->gtAlgorithmMap().begin(); algo!=menu->gtAlgorithmMap().end(); ++algo) {
	if (_Debug) std::cout << "Name: " << (algo->second).algoName() << " Alias: " << (algo->second).algoAlias() << std::endl;
        int itrig = (algo->second).algoBitNumber();
	//        algoBitToName[itrig] = TString( (algo->second).algoName() );
	algoBitToName[itrig] = TString( (algo->second).algoAlias() );
        HltTree->Branch(algoBitToName[itrig],l1flag+itrig,algoBitToName[itrig]+"/I");
        HltTree->Branch(algoBitToName[itrig]+"_Prescl",l1Prescl+itrig,algoBitToName[itrig]+"_Prescl/I");
	if (_OR_BXes)
	  HltTree->Branch(algoBitToName[itrig]+"_5bx",l1flag5Bx+itrig,algoBitToName[itrig]+"_5bx/I");
      }

      // Book branches for tech bits
      for (CItAlgo techTrig = menu->gtTechnicalTriggerMap().begin(); techTrig != menu->gtTechnicalTriggerMap().end(); ++techTrig) {
        int itrig = (techTrig->second).algoBitNumber();
	techBitToName[itrig] = TString( (techTrig->second).algoName() );
	if (_Debug) std::cout << "tech bit " << itrig << ": " << techBitToName[itrig] << " " << std::endl;
	HltTree->Branch(techBitToName[itrig],l1techflag+itrig,techBitToName[itrig]+"/I");
        HltTree->Branch(techBitToName[itrig]+"_Prescl",l1techPrescl+itrig,techBitToName[itrig]+"_Prescl/I");
	if (_OR_BXes)
	  HltTree->Branch(techBitToName[itrig]+"_5bx",l1techflag5Bx+itrig,techBitToName[itrig]+"_5bx/I");
      }
    }

    std::string triggerAlgTechTrig = "PhysicsAlgorithms";
    for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {     
      // ...Fill the corresponding accepts in branch-variables
      l1flag[iBit] = gtDecisionWord[iBit];

      std::string l1triggername= std::string (algoBitToName[iBit]);
      l1Prescl[iBit] = l1GtUtils.prescaleFactor(iEvent, 
					       l1triggername,
					       iErrorCode);
      
      if (_Debug) std::cout << "L1 TD: "<<iBit<<" "<<algoBitToName[iBit]<<" "
			    << gtDecisionWord[iBit]<<" "
			    << l1Prescl[iBit] << std::endl;

    }

    triggerAlgTechTrig = "TechnicalTriggers";
    for (unsigned int iBit = 0; iBit < numberTechnicalTriggerBits; ++iBit) {
      l1techflag[iBit] = (int) technicalTriggerWordBeforeMask.at(iBit);

      std::string l1triggername= std::string (techBitToName[iBit]);
      l1techPrescl[iBit] = l1GtUtils.prescaleFactor(iEvent, 
					       l1triggername,
					       iErrorCode);

      if (_Debug) std::cout << "L1 TD: "<<iBit<<" "<<techBitToName[iBit]<<" "
			    << l1techflag[iBit]<<" "
			    << l1Prescl[iBit] << std::endl;

    }

    if (_OR_BXes){
      // look at all 5 bx window in case gt timing is off
      // get Field Decision Logic
      std::vector<DecisionWord> m_gtDecisionWord5Bx;
      std::vector<TechnicalTriggerWord> m_gtTechDecisionWord5Bx;
      std::vector<int> m_ibxn;

      const std::vector<L1GtFdlWord> &m_gtFdlWord(L1GTRR->gtFdlVector());
      for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
	   itBx != m_gtFdlWord.end(); ++itBx) {
	if (_Debug && L1EvtCnt==0) std::cout << "bx: " << (*itBx).bxInEvent() << " ";
	m_gtDecisionWord5Bx.push_back((*itBx).gtDecisionWord());
	m_gtTechDecisionWord5Bx.push_back((*itBx).gtTechnicalTriggerWord());
      }
      // --- Fill algo bits ---
      for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {     
	// ...Fill the corresponding accepts in branch-variables
	if (_Debug) std::cout << std::endl << " L1 TD: "<<iBit<<" "<<algoBitToName[iBit]<<" ";
	int result=0;
	int bitword=0; 
	for (unsigned int jbx=0; jbx<m_gtDecisionWord5Bx.size(); ++jbx) {
	  if (_Debug) std::cout << m_gtDecisionWord5Bx[jbx][iBit]<< " ";
	  result += m_gtDecisionWord5Bx[jbx][iBit];
	  if (m_gtDecisionWord5Bx[jbx][iBit]>0) bitword |= 1 << jbx;
	}
	if (_Debug && result>1) {std::cout << "5BxOr=" << result << "  Bitword= "<< bitword <<std::endl;
	  std::cout << "Unpacking: " ;
	  for (int i = 0; i<UnpackBxInEvent ; ++i){
	    bool bitOn=bitword & (1 << i);
	    std::cout << bitOn << " ";
	  }
	  std::cout << "\n";
	}
	l1flag5Bx[iBit] = bitword;
      }
      // --- Fill tech bits ---
      for (unsigned int iBit = 0; iBit < m_gtTechDecisionWord5Bx[2].size(); ++iBit) {     
	// ...Fill the corresponding accepts in branch-variables
	if (_Debug) std::cout << std::endl << " L1 TD: "<<iBit<<" "<<techBitToName[iBit]<<" ";
	int result=0;
	int bitword=0;       
	for (unsigned int jbx=0; jbx<m_gtTechDecisionWord5Bx.size(); ++jbx) {
	  if (_Debug) std::cout << m_gtTechDecisionWord5Bx[jbx][iBit]<< " ";
	  result += m_gtTechDecisionWord5Bx[jbx][iBit];
	  if (m_gtTechDecisionWord5Bx[jbx][iBit]>0) bitword |= 1 << jbx;
	}
	if (_Debug && result>1) {std::cout << "5BxOr=" << result << "  Bitword= "<< bitword  << std::endl;
	  std::cout << "Unpacking: " ;
	  for (int i = 0; i<UnpackBxInEvent ; ++i){
	    bool bitOn=bitword & (1 << i);
	    std::cout << bitOn << " ";
	  }
	  std::cout << "\n";
	}
	l1techflag5Bx[iBit] = bitword;
      }
    } // end of OR_BX

    L1EvtCnt++;
  }
  else {
    if (_Debug) std::cout << "%HLTInfo -- No L1 GT ReadoutRecord " << std::endl;
  }

  //
  // LSB for feature bits = 0.125 GeV.
  // The default LSB for the ring sums is 0.5 GeV.
  
  if (gctBitCounts.isValid()) {
    L1GctHFBitCountsCollection::const_iterator bitCountItr;
    for (bitCountItr=gctBitCounts->begin(); bitCountItr!=gctBitCounts->end(); ++bitCountItr) { 
      if (bitCountItr->bx()==0){ // select in-time beam crossing
	l1hfTowerCountPositiveEtaRing1=bitCountItr->bitCount(0);
	l1hfTowerCountNegativeEtaRing1=bitCountItr->bitCount(1);
	l1hfTowerCountPositiveEtaRing2=bitCountItr->bitCount(2);
	l1hfTowerCountNegativeEtaRing2=bitCountItr->bitCount(3);
      }
    }
  } else {
    if (_Debug) std::cout << "%HLTInfo -- No L1 Gct HF BitCounts" << std::endl;
  }

  if (gctRingSums.isValid()) {
    L1GctHFRingEtSumsCollection::const_iterator ringSumsItr;
    for (ringSumsItr=gctRingSums->begin(); ringSumsItr!=gctRingSums->end(); ++ringSumsItr) { 
      if (ringSumsItr->bx()==0){ // select in-time beam crossing
	l1hfRing1EtSumPositiveEta=ringSumsItr->etSum(0);
	l1hfRing1EtSumNegativeEta=ringSumsItr->etSum(1);
	l1hfRing2EtSumPositiveEta=ringSumsItr->etSum(2);
	l1hfRing2EtSumNegativeEta=ringSumsItr->etSum(3);
      }
    }
  } else {
    if (_Debug) std::cout << "%HLTInfo -- No L1 Gct HF RingSums" << std::endl;
  }

  if (_Debug) std::cout << "%HLTInfo -- Done with routine" << std::endl;
}
