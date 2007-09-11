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

HLTInfo::HLTInfo() {

  //set parameter defaults 
  _Debug=false;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTInfo::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  edm::ParameterSet myHltParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  vector<std::string> parameterNames = myHltParams.getParameterNames() ;
  
  for ( vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if ( (*iParam) == "Debug" ) _Debug =  myHltParams.getParameter<bool>( *iParam );
  }

  HltEvtCnt = 0;
  const int kMaxTrigFlag = 10000;
  trigflag = new int[kMaxTrigFlag];
  L1EvtCnt = 0;
  const int kMaxL1Flag = 10000;
  l1flag = new int[kMaxL1Flag];
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
  const int kMaxL1ExtTau = 10000;
  l1exttauet = new float[kMaxL1ExtTau];
  l1exttaue = new float[kMaxL1ExtTau];
  l1exttaueta = new float[kMaxL1ExtTau];
  l1exttauphi = new float[kMaxL1ExtTau];

  // HLT-specific branches of the tree 
  HltTree->Branch("NobjHltPart",&nhltpart,"NobjHltPart/I");
  HltTree->Branch("HLTPartPt",hltppt,"HLTPartPt[NobjHltPart]/F");
  HltTree->Branch("HLTPartEta",hltpeta,"HLTPartEta[NobjHltPart]/F");

  HltTree->Branch("NobjL1ExtIsolEm",&nl1extiem,"NobjL1ExtIsolEm/I");
  HltTree->Branch("L1ExtIsolEmEt",l1extiemet,"L1ExtIsolEmEt[NobjL1ExtIsolEm]/F");
  HltTree->Branch("L1ExtIsolEmE",l1extieme,"L1ExtIsolEmE[NobjL1ExtIsolEm]/F");
  HltTree->Branch("L1ExtIsolEmEta",l1extiemeta,"L1ExtIsolEmEta[NobjL1ExtIsolEm]/F");
  HltTree->Branch("L1ExtIsolEmPhi",l1extiemphi,"L1ExtIsolEmPhi[NobjL1ExtIsolEm]/F");

  HltTree->Branch("NobjL1ExtNIsolEm",&nl1extnem,"NobjL1ExtNIsolEm/I");
  HltTree->Branch("L1ExtNIsolEmEt",l1extnemet,"L1ExtNIsolEmEt[NobjL1ExtNIsolEm]/F");
  HltTree->Branch("L1ExtNIsolEmE",l1extneme,"L1ExtNIsolEmE[NobjL1ExtNIsolEm]/F");
  HltTree->Branch("L1ExtNIsolEmEta",l1extnemeta,"L1ExtNIsolEmEta[NobjL1ExtNIsolEm]/F");
  HltTree->Branch("L1ExtNIsolEmPhi",l1extnemphi,"L1ExtNIsolEmPhi[NobjL1ExtNIsolEm]/F");

  HltTree->Branch("NobjL1ExtMu",&nl1extmu,"NobjL1ExtMu/I");
  HltTree->Branch("L1ExtMuPt",l1extmupt,"L1ExtMuPt[NobjL1ExtMu]/F");
  HltTree->Branch("L1ExtMuE",l1extmue,"L1ExtMuE[NobjL1ExtMu]/F");
  HltTree->Branch("L1ExtMuEta",l1extmueta,"L1ExtMuEta[NobjL1ExtMu]/F");
  HltTree->Branch("L1ExtMuPhi",l1extmuphi,"L1ExtMuPhi[NobjL1ExtMu]/F");
  HltTree->Branch("L1ExtMuIsol",l1extmuiso,"L1ExtMuIsol[NobjL1ExtMu]/I");
  HltTree->Branch("L1ExtMuMip",l1extmumip,"L1ExtMuMip[NobjL1ExtMu]/I");
  HltTree->Branch("NobjL1ExtCenJet",&nl1extjetc,"NobjL1ExtCenJet/I");
  HltTree->Branch("L1ExtCenJetEt",l1extjtcet,"L1ExtCenJetEt[NobjL1ExtCenJet]/F");
  HltTree->Branch("L1ExtCenJetE",l1extjtce,"L1ExtCenJetE[NobjL1ExtCenJet]/F");
  HltTree->Branch("L1ExtCenJetEta",l1extjtceta,"L1ExtCenJetEta[NobjL1ExtCenJet]/F");
  HltTree->Branch("L1ExtCenJetPhi",l1extjtcphi,"L1ExtCenJetPhi[NobjL1ExtCenJet]/F");
  HltTree->Branch("NobjL1ExtForJet",&nl1extjetf,"NobjL1ExtForJet/I");
  HltTree->Branch("L1ExtForJetEt",l1extjtfet,"L1ExtForJetEt[NobjL1ExtForJet]/F");
  HltTree->Branch("L1ExtForJetE",l1extjtfe,"L1ExtForJetE[NobjL1ExtForJet]/F");
  HltTree->Branch("L1ExtForJetEta",l1extjtfeta,"L1ExtForJetEta[NobjL1ExtForJet]/F");
  HltTree->Branch("L1ExtForJetPhi",l1extjtfphi,"L1ExtForJetPhi[NobjL1ExtForJet]/F");
  HltTree->Branch("NobjL1ExtTau",&nl1exttau,"NobjL1ExtTau/I");
  HltTree->Branch("L1ExtTauEt",l1exttauet,"L1ExtTauEt[NobjL1ExtTau]/F");
  HltTree->Branch("L1ExtTauE",l1exttaue,"L1ExtTauE[NobjL1ExtTau]/F");
  HltTree->Branch("L1ExtTauEta",l1exttaueta,"L1ExtTauEta[NobjL1ExtTau]/F");
  HltTree->Branch("L1ExtTauPhi",l1exttauphi,"L1ExtTauPhi[NobjL1ExtTau]/F");
  HltTree->Branch("L1ExtMet",&met,"L1ExtMet/F");
  HltTree->Branch("L1ExtMetPhi",&metphi,"L1ExtMetPhi/F");
  HltTree->Branch("L1ExtMetTot",&mettot,"L1ExtMetTot/F");
  HltTree->Branch("L1ExtMetHad",&methad,"L1ExtMetHad/F");

}

/* **Analyze the event** */
void HLTInfo::analyze(const HLTFilterObjectWithRefs& hltobj,
		      const edm::TriggerResults& hltresults,
		      const l1extra::L1EmParticleCollection& L1ExtEmIsol,
		      const l1extra::L1EmParticleCollection& L1ExtEmNIsol,
		      const l1extra::L1MuonParticleCollection& L1ExtMu,
		      const l1extra::L1JetParticleCollection& L1ExtJetC,
		      const l1extra::L1JetParticleCollection& L1ExtJetF,
		      const l1extra::L1JetParticleCollection& L1ExtTau,
		      const l1extra::L1EtMissParticle& L1ExtMet,
		      const l1extra::L1ParticleMapCollection& L1MapColl,
		      TTree* HltTree) {

//   std::cout << " Beginning HLTInfo " << std::endl;


  /////////// Analyzing HLT Trigger Results (TriggerResults) //////////

  if (&hltresults) {
    int ntrigs=hltresults.size();
    if (ntrigs==0){std::cout << "%HLTInfo -- No trigger name given in TriggerResults of the input " << std::endl;}

    triggerNames_.init(hltresults);

    // 1st event : Book as many branches as trigger paths provided in the input...
    if (HltEvtCnt==0){
      for (int itrig = 0; itrig != ntrigs; ++itrig){
	TString trigName = triggerNames_.triggerName(itrig);
	HltTree->Branch("TRIGG_"+trigName,trigflag+itrig,"TRIGG_"+trigName+"/I");
      }
      HltEvtCnt++;
    }
    // ...Fill the corresponding accepts in branch-variables
    for (int itrig = 0; itrig != ntrigs; ++itrig){

      string trigName=triggerNames_.triggerName(itrig);
      bool accept = hltresults.accept(itrig);

      if (accept){trigflag[itrig] = 1;}
      else {trigflag[itrig] = 0;}

      if (_Debug){
	if (_Debug) std::cout << "%HLTInfo --  Number of HLT Triggers: " << ntrigs << std::endl;
	std::cout << "%HLTInfo --  HLTTrigger(" << itrig << "): " << trigName << " = " << accept << std::endl;
      }

    }
  }
  else { if (_Debug) std::cout << "%HLTInfo -- No Trigger Result" << std::endl;}

  /////////// Analyzing HLT Objects (HLTFilterObjectsWithRefs) //////////
 
  int mod=-1,path=-1,npart=-1;

  if (&hltobj) {
    mod = hltobj.module();
    path = hltobj.path();
    npart = hltobj.size();
    nhltpart = npart;
    for (int ipart = 0; ipart != npart; ++ipart){
      const edm::RefToBase<Candidate> ref_ = hltobj.getParticleRef(ipart);
      hltppt[ipart] = ref_->pt();
      hltpeta[ipart] = ref_->eta();
    }

    if (_Debug){
      std::cout << "%HLTInfo --  HLTobj module: " << mod << "   path: " << path << "   Npart:" << npart << std::endl;
    }

  }
  else {
    nhltpart = 0;
    if (_Debug) std::cout << "%HLTInfo -- No HLT Object" << std::endl;
  }

  /////////// Analyzing L1Extra (from MC Truth) objects //////////

  if (&L1ExtEmIsol) {
    nl1extiem = L1ExtEmIsol.size();
    l1extra::L1EmParticleCollection myl1iems;
    myl1iems=L1ExtEmIsol;
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
  if (&L1ExtEmNIsol) {
    nl1extnem = L1ExtEmNIsol.size();
    l1extra::L1EmParticleCollection myl1nems;
    myl1nems=L1ExtEmNIsol;
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

  if (&L1ExtMu) {
    nl1extmu = L1ExtMu.size();
    l1extra::L1MuonParticleCollection myl1mus;
    myl1mus=L1ExtMu;
    std::sort(myl1mus.begin(),myl1mus.end(),PtGreater());
    int il1exmu = 0;
    for (l1extra::L1MuonParticleCollection::const_iterator muItr = myl1mus.begin(); muItr != myl1mus.end(); ++muItr) {
      l1extmupt[il1exmu] = muItr->pt();
      l1extmue[il1exmu] = muItr->energy();
      l1extmueta[il1exmu] = muItr->eta();
      l1extmuphi[il1exmu] = muItr->phi();
      l1extmuiso[il1exmu] = muItr->isIsolated(); // = 1 for Isolated ?
      l1extmumip[il1exmu] = muItr->isMip(); // = 1 for Mip ?
      il1exmu++;
    }
  }
  else {
    nl1extmu = 0;
    if (_Debug) std::cout << "%HLTInfo -- No L1 MU object" << std::endl;
  }

  if (&L1ExtJetC) {
    nl1extjetc = L1ExtJetC.size();
    l1extra::L1JetParticleCollection myl1jetsc;
    myl1jetsc=L1ExtJetC;
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
  if (&L1ExtJetF) {
    nl1extjetf = L1ExtJetF.size();
    l1extra::L1JetParticleCollection myl1jetsf;
    myl1jetsf=L1ExtJetF;
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

  if (&L1ExtTau) {
    nl1exttau = L1ExtTau.size();
    l1extra::L1JetParticleCollection myl1taus;
    myl1taus=L1ExtTau;
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

  if (&L1ExtMet) {
    met = L1ExtMet.energy();
    metphi = L1ExtMet.phi();
    mettot = L1ExtMet.etTotal();
    methad = L1ExtMet.etHad();
  }
  else {
    if (_Debug) std::cout << "%HLTInfo -- No L1 MET object" << std::endl;
  }

  if (&L1MapColl) {

    // 1st event : Book as many branches as trigger paths provided in the input...
    if (L1EvtCnt==0){
      for (int itrig = 0; itrig != l1extra::L1ParticleMap::kNumOfL1TriggerTypes; ++itrig){
	const l1extra::L1ParticleMap& map = ( L1MapColl )[ itrig ] ;
	TString trigName = map.triggerName();
	HltTree->Branch(trigName,l1flag+itrig,trigName+"/I");
      }
      L1EvtCnt++;
    }
    // ...Fill the corresponding accepts in branch-variables
    for (int itrig = 0; itrig != l1extra::L1ParticleMap::kNumOfL1TriggerTypes; ++itrig){
      const l1extra::L1ParticleMap& map = ( L1MapColl )[ itrig ] ;
      l1flag[itrig] = map.triggerDecision();
    }
   
  }
  else {
    if (_Debug) std::cout << "%HLTInfo -- No L1 Map Collection" << std::endl;
  }

}
