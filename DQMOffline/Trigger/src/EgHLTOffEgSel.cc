#include "DQMOffline/Trigger/interface/EgHLTOffEgSel.h"

#include "DQMOffline/Trigger/interface/EgHLTEgCutCodes.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"
#include "DQMOffline/Trigger/interface/EgHLTOffPho.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace egHLT;

void OffEgSel::setup(const edm::ParameterSet& iConfig)
{ 
  ebCutValues_.setup(iConfig.getParameter<edm::ParameterSet>("barrel"));
  eeCutValues_.setup(iConfig.getParameter<edm::ParameterSet>("endcap"));
}


int OffEgSel::getCutCode(const OffEle& ele,int cutMask)const
{
  if(ele.detEta()<1.5) return getCutCode(ele,ebCutValues_,cutMask);
  else return getCutCode(ele,eeCutValues_,cutMask);
}

int OffEgSel::getCutCode(const OffEle& ele,const EgCutValues& cuts,int cutMask)
{ 
  int cutCode = 0x0;
  //kinematic cuts
  if(ele.et()< cuts.minEt) cutCode |= EgCutCodes::ET;
  if(fabs(ele.etaSC())< cuts.minEta || fabs(ele.etaSC())>cuts.maxEta) cutCode |= EgCutCodes::DETETA;
  if(ele.classification()==40) cutCode |= EgCutCodes::CRACK;
  //track cuts
  if(fabs(ele.dEtaIn()) > cuts.maxDEtaIn ) cutCode |=EgCutCodes::DETAIN;
  if(fabs(ele.dPhiIn()) > cuts.maxDPhiIn ) cutCode |=EgCutCodes::DPHIIN;
  if(ele.invEOverInvP() > cuts.maxInvEInvP) cutCode |= EgCutCodes::INVEINVP;
  //supercluster cuts
  if(ele.hOverE()> cuts.maxHadem) cutCode |= EgCutCodes::HADEM;
  if(ele.scSigmaIEtaIEta()>cuts.maxSigmaIEtaIEta) cutCode |= EgCutCodes::SIGMAIETAIETA;
  if(ele.r9()<cuts.minR9) cutCode |= EgCutCodes::R9;
  
  //std isolation cuts
  if(ele.isolEm()>( cuts.isolEmConstTerm + cuts.isolEmGradTerm*(ele.et()<cuts.isolEmGradStart ? 0. : (ele.et()-cuts.isolEmGradStart)))) cutCode |=EgCutCodes::ISOLEM;
  if(ele.isolHad()> (cuts.isolHadConstTerm + cuts.isolHadGradTerm*(ele.et()<cuts.isolHadGradStart ? 0. : (ele.et()-cuts.isolHadGradStart)))) cutCode |=EgCutCodes::ISOLHAD;
  if(ele.isolPtTrks() > (cuts.isolPtTrksConstTerm + cuts.isolPtTrksGradTerm*(ele.et()<cuts.isolPtTrksGradStart ? 0. : (ele.et()-cuts.isolPtTrksGradStart))))cutCode |=EgCutCodes::ISOLPTTRKS; 
  //ele Nr trks not defined, assume it passes
  //hlt isolation cuts
  if(ele.hltIsolTrksEle() > cuts.maxHLTIsolTrksEle) cutCode |=EgCutCodes::HLTISOLTRKSELE;
  if(ele.hltIsolTrksPho() > cuts.maxHLTIsolTrksPho) cutCode |=EgCutCodes::HLTISOLTRKSPHO;
  
  if(ele.et()==0 || (ele.hltIsolHad() > cuts.maxHLTIsolHad && ele.hltIsolHad()/ele.et() > cuts.maxHLTIsolHadOverEt &&
		     ele.hltIsolHad()/ele.et()/ele.et() > cuts.maxHLTIsolHadOverEt2)) cutCode |=EgCutCodes::HLTISOLHAD;

  return (cutCode & cuts.cutMask & cutMask) ;
}

int OffEgSel::getCutCode(const OffPho& pho,int cutMask)const
{
  if(pho.detEta()<1.5) return getCutCode(pho,ebCutValues_,cutMask);
  else return getCutCode(pho,eeCutValues_,cutMask);
}

//photons automatically fail any track cut
int OffEgSel::getCutCode(const OffPho& pho,const EgCutValues& cuts,int cutMask)
{ 
  int cutCode = 0x0;
  //kinematic cuts
  if(pho.et()< cuts.minEt) cutCode |= EgCutCodes::ET;
  if(fabs(pho.etaSC())< cuts.minEta || fabs(pho.etaSC())>cuts.maxEta) cutCode |= EgCutCodes::DETETA;
  //if(pho.classification()==40) cutCode |= EgCutCodes::CRACK;
  //track cuts (all fail)
  cutCode |=EgCutCodes::DETAIN;
  cutCode |=EgCutCodes::DPHIIN;
  cutCode |=EgCutCodes::INVEINVP;
  //supercluster cuts
  if(pho.hOverE()> cuts.maxHadem) cutCode |= EgCutCodes::HADEM;
  if(pho.scSigmaIEtaIEta()>cuts.maxSigmaIEtaIEta) cutCode |= EgCutCodes::SIGMAIETAIETA; 
  if(pho.r9()<cuts.minR9) cutCode |= EgCutCodes::R9;
  //std isolation cuts
  if(pho.isolEm()>( cuts.isolEmConstTerm + cuts.isolEmGradTerm*(pho.et()<cuts.isolEmGradStart ? 0. : (pho.et()-cuts.isolEmGradStart)))) cutCode |=EgCutCodes::ISOLEM;
  if(pho.isolHad()> (cuts.isolHadConstTerm + cuts.isolHadGradTerm*(pho.et()<cuts.isolHadGradStart ? 0. : (pho.et()-cuts.isolHadGradStart)))) cutCode |=EgCutCodes::ISOLHAD;
  if(pho.isolPtTrks() > (cuts.isolPtTrksConstTerm + cuts.isolPtTrksGradTerm*(pho.et()<cuts.isolPtTrksGradStart ? 0. : (pho.et()-cuts.isolPtTrksGradStart))))cutCode |=EgCutCodes::ISOLPTTRKS;
  if(pho.isolNrTrks() > cuts.isolNrTrksConstTerm) cutCode |=EgCutCodes::ISOLNRTRKS;

  //hlt isolation cuts
  cutCode |=EgCutCodes::HLTISOLTRKSELE; //automatically fails ele track isolation
  if(pho.hltIsolTrks() > cuts.maxHLTIsolTrksPho) cutCode |=EgCutCodes::HLTISOLTRKSPHO;
  if(pho.et()==0 || (pho.hltIsolHad() > cuts.maxHLTIsolHad && pho.hltIsolHad()/pho.et() > cuts.maxHLTIsolHadOverEt &&
		     pho.hltIsolHad()/pho.et()/pho.et() > cuts.maxHLTIsolHadOverEt2)) cutCode |=EgCutCodes::HLTISOLHAD;

  return (cutCode & cuts.cutMask & cutMask) ;
}
