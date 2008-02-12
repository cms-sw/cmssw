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
  l1extmufor = new int[kMaxL1ExtMu];
  l1extmurpc = new int[kMaxL1ExtMu];
  l1extmuqul = new int[kMaxL1ExtMu];
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
//   HltTree->Branch("NHltPart",&nhltpart,"NHltPart/I");
//   HltTree->Branch("HLTPartPt",hltppt,"HLTPartPt[NHltPart]/F");
//   HltTree->Branch("HLTPartEta",hltpeta,"HLTPartEta[NHltPart]/F");

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
  HltTree->Branch("NL1Tau",&nl1exttau,"NL1Tau/I");
  HltTree->Branch("L1TauEt",l1exttauet,"L1TauEt[NL1Tau]/F");
  HltTree->Branch("L1TauE",l1exttaue,"L1TauE[NL1Tau]/F");
  HltTree->Branch("L1TauEta",l1exttaueta,"L1TauEta[NL1Tau]/F");
  HltTree->Branch("L1TauPhi",l1exttauphi,"L1TauPhi[NL1Tau]/F");
  HltTree->Branch("L1Met",&met,"L1Met/F");
  HltTree->Branch("L1MetPhi",&metphi,"L1MetPhi/F");
  HltTree->Branch("L1MetTot",&mettot,"L1MetTot/F");
  HltTree->Branch("L1MetHad",&methad,"L1MetHad/F");

}

/* **Analyze the event** */
void HLTInfo::analyze(/*const HLTFilterObjectWithRefs& hltobj,*/
		      const edm::TriggerResults& hltresults,
		      const l1extra::L1EmParticleCollection& L1ExtEmIsol,
		      const l1extra::L1EmParticleCollection& L1ExtEmNIsol,
		      const l1extra::L1MuonParticleCollection& L1ExtMu,
		      const l1extra::L1JetParticleCollection& L1ExtJetC,
		      const l1extra::L1JetParticleCollection& L1ExtJetF,
		      const l1extra::L1JetParticleCollection& L1ExtTau,
		      const l1extra::L1EtMissParticleCollection& L1ExtMet,
//		      const l1extra::L1ParticleMapCollection& L1MapColl,
		      const L1GlobalTriggerReadoutRecord& L1GTRR,
		      const L1GlobalTriggerObjectMapRecord& L1GTOMRec,
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
// 	HltTree->Branch("TRIGG_"+trigName,trigflag+itrig,"TRIGG_"+trigName+"/I");
	HltTree->Branch(trigName,trigflag+itrig,trigName+"/I");
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
 
//   int mod=-1,path=-1,npart=-1;

//   if (&hltobj) {
//     mod = hltobj.module();
//     path = hltobj.path();
//     npart = hltobj.size();
//     nhltpart = npart;
//     for (int ipart = 0; ipart != npart; ++ipart){
//       const edm::RefToBase<Candidate> ref_ = hltobj.getParticleRef(ipart);
//       hltppt[ipart] = ref_->pt();
//       hltpeta[ipart] = ref_->eta();
//     }

//     if (_Debug){
//       std::cout << "%HLTInfo --  HLTobj module: " << mod << "   path: " << path << "   Npart:" << npart << std::endl;
//     }

//   }
//   else {
//     nhltpart = 0;
//     if (_Debug) std::cout << "%HLTInfo -- No HLT Object" << std::endl;
//   }

  /////////// Analyzing L1Extra objects //////////

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
      l1extmufor[il1exmu] = muItr->isForward();
      l1extmurpc[il1exmu] = muItr->isRPC();
      L1MuGMTExtendedCand gmtCand = muItr->gmtMuonCand();
      l1extmuqul[il1exmu] = gmtCand.quality(); // Muon quality as defined in the GT
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
    met = L1ExtMet.begin()->energy();
    metphi = L1ExtMet.begin()->phi();
    mettot = L1ExtMet.begin()->etTotal();
    methad = L1ExtMet.begin()->etHad();
  }
  else {
    if (_Debug) std::cout << "%HLTInfo -- No L1 MET object" << std::endl;
  }

  /* comment out full block: uses the obsolete l1ExtraParticleMap|Collection

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
    if (_Debug) std::cout << "%HLTInfo -- No [obsolete] L1 Map Collection" << std::endl;
  }
  */

  TString algoBitToName[128];
  DecisionWord gtDecisionWord = L1GTRR.decisionWord();
  const unsigned int numberTriggerBits(gtDecisionWord.size());
  // 1st event : Book as many branches as trigger paths provided in the input...
  if ((&L1GTRR) && (&L1GTOMRec)) {  
    if (L1EvtCnt==0){
      // get ObjectMaps from ObjectMapRecord
      const std::vector<L1GlobalTriggerObjectMap>& objMapVec =  L1GTOMRec.gtObjectMap();
      // 1st event : Book as many branches as trigger paths provided in the input...
      for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
	   itMap != objMapVec.end(); ++itMap) {
	// Get trigger bits
	int itrig = (*itMap).algoBitNumber();
	// Get trigger names
	algoBitToName[itrig] = TString( (*itMap).algoName() );
	
	HltTree->Branch(algoBitToName[itrig],l1flag+itrig,algoBitToName[itrig]+"/I");
      }
      L1EvtCnt++;
    }
    for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {	
      // ...Fill the corresponding accepts in branch-variables
      l1flag[iBit] = gtDecisionWord[iBit];
      //std::cout << "L1 TD: "<<iBit<<" "<<algoBitToName[iBit]<<" "<<gtDecisionWord[iBit]<< std::endl;
    }
  }
  else {
    if (_Debug) std::cout << "%HLTInfo -- No L1 GT ReadoutRecord or ObjectMapRecord" << std::endl;
  }

}
