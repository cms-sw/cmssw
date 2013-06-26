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
  if(fabs(ele.detEta())<1.5) return getCutCode(ele,ebCutValues_,cutMask);
  else return getCutCode(ele,eeCutValues_,cutMask);
}

int OffEgSel::getCutCode(const OffEle& ele,const EgCutValues& cuts,int cutMask)
{ 
  int cutCode = 0x0;
  //kinematic cuts
  if(ele.et() < cuts.minEt) cutCode |= EgCutCodes::ET;
  if(fabs(ele.etaSC()) < cuts.minEta || fabs(ele.etaSC()) > cuts.maxEta) cutCode |= EgCutCodes::DETETA;
  if(ele.isGap()) cutCode |= EgCutCodes::CRACK;
  //track cuts
  if(fabs(ele.dEtaIn()) > cuts.maxDEtaIn ) cutCode |=EgCutCodes::DETAIN;
  if(fabs(ele.dPhiIn()) > cuts.maxDPhiIn ) cutCode |=EgCutCodes::DPHIIN;
  if(ele.invEInvP() > cuts.maxInvEInvP) cutCode |= EgCutCodes::INVEINVP;
  //supercluster cuts
  if(ele.hOverE() > cuts.maxHadem && ele.hOverE()*ele.caloEnergy() > cuts.maxHadEnergy) cutCode |= EgCutCodes::HADEM;
  if(ele.sigmaIEtaIEta() > cuts.maxSigmaIEtaIEta) cutCode |= EgCutCodes::SIGMAIETAIETA;
  if(ele.sigmaEtaEta() > cuts.maxSigmaEtaEta) cutCode |= EgCutCodes::SIGMAETAETA;
  //---Morse-------
  //if(ele.r9()<cuts.minR9) cutCode |= EgCutCodes::R9;
  if(ele.r9() < cuts.minR9) cutCode |= EgCutCodes::MINR9;
  if(ele.r9() > cuts.maxR9) cutCode |= EgCutCodes::MAXR9;
  //----------------
  
  //std isolation cuts
  if(ele.isolEm()>( cuts.isolEmConstTerm + cuts.isolEmGradTerm*(ele.et()<cuts.isolEmGradStart ? 0. : (ele.et()-cuts.isolEmGradStart)))) cutCode |=EgCutCodes::ISOLEM;
  if(ele.isolHad()> (cuts.isolHadConstTerm + cuts.isolHadGradTerm*(ele.et()<cuts.isolHadGradStart ? 0. : (ele.et()-cuts.isolHadGradStart)))) cutCode |=EgCutCodes::ISOLHAD;
  if(ele.isolPtTrks() > (cuts.isolPtTrksConstTerm + cuts.isolPtTrksGradTerm*(ele.et()<cuts.isolPtTrksGradStart ? 0. : (ele.et()-cuts.isolPtTrksGradStart))))cutCode |=EgCutCodes::ISOLPTTRKS; 
  //ele Nr trks not defined, assume it passes
  //hlt isolation cuts
  if(ele.et()<=0.){//even it if et<=0, we give it a shot at passing isolation. Note this should be an impossible case
    if(ele.hltIsolTrksEle() > cuts.maxHLTIsolTrksEle) cutCode |=EgCutCodes::HLTISOLTRKSELE;
    if(ele.hltIsolTrksPho() > cuts.maxHLTIsolTrksPho) cutCode |=EgCutCodes::HLTISOLTRKSPHO;
    if(ele.hltIsolHad() > cuts.maxHLTIsolHad) cutCode |=EgCutCodes::HLTISOLHAD;
    if(ele.hltIsolEm() > cuts.maxHLTIsolEm) cutCode |=EgCutCodes::HLTISOLEM;
  }else{ 
    if(ele.hltIsolTrksEle() > cuts.maxHLTIsolTrksEle && ele.hltIsolTrksEle()/ele.et() > cuts.maxHLTIsolTrksEleOverPt &&
       ele.hltIsolTrksEle()/ele.et()/ele.et() > cuts.maxHLTIsolTrksEleOverPt2 ) cutCode |=EgCutCodes::HLTISOLTRKSELE;
    if(ele.hltIsolTrksPho() > cuts.maxHLTIsolTrksPho && ele.hltIsolTrksPho()/ele.et() > cuts.maxHLTIsolTrksPhoOverPt &&
       ele.hltIsolTrksPho()/ele.et()/ele.et() > cuts.maxHLTIsolTrksPhoOverPt2 ) cutCode |=EgCutCodes::HLTISOLTRKSPHO;
    if(ele.hltIsolHad() > cuts.maxHLTIsolHad && ele.hltIsolHad()/ele.et() > cuts.maxHLTIsolHadOverEt &&
       ele.hltIsolHad()/ele.et()/ele.et() > cuts.maxHLTIsolHadOverEt2) cutCode |=EgCutCodes::HLTISOLHAD;
    if(ele.hltIsolEm() > cuts.maxHLTIsolEm && ele.hltIsolEm()/ele.et() > cuts.maxHLTIsolEmOverEt &&
       ele.hltIsolEm()/ele.et()/ele.et() > cuts.maxHLTIsolEmOverEt2) cutCode |=EgCutCodes::HLTISOLEM;
  }
 
 
  //cuts on CTF track, HLT tracking doesnt handle poor quaility tracks
  if(ele.validCTFTrack()){
    if(!(ele.ctfTrkOuterRadius() >= cuts.minCTFTrkOuterRadius && //note I'm NOTing the result of the AND
	 ele.ctfTrkInnerRadius() <= cuts.maxCTFTrkInnerRadius &&
	 
	 ele.ctfTrkHitsFound() >= cuts.minNrCTFTrkHits && 
	 ele.ctfTrkHitsLost() <= cuts.maxNrCTFTrkHitsLost)) cutCode |=EgCutCodes::CTFTRACK; //the next line can also set this bit
    if(cuts.requirePixelHitsIfOuterInOuter){
      DetId innerDetId(ele.ctfTrack()->extra()->innerDetId());
      DetId outerDetId(ele.ctfTrack()->extra()->outerDetId());
    
      if(outerDetId.subdetId()>=5 && innerDetId.subdetId()>=3) cutCode |=EgCutCodes::CTFTRACK; //1,2 = pixel, 3,4,5,6 sistrip
    }
    // std::cout <<"eta "<<ele.detEta()<<" max inner "<<cuts.maxCTFTrkInnerRadius<<" inner "<<ele.ctfTrkInnerRadius()<<std::endl;
  }else cutCode |=EgCutCodes::CTFTRACK;

  if(fabs(ele.hltDEtaIn()) > cuts.maxHLTDEtaIn) cutCode |=EgCutCodes::HLTDETAIN;
  if(fabs(ele.hltDPhiIn()) > cuts.maxHLTDPhiIn) cutCode |=EgCutCodes::HLTDPHIIN;
  if(fabs(ele.hltInvEInvP()) > cuts.maxHLTInvEInvP) cutCode |=EgCutCodes::HLTINVEINVP;

  return (cutCode & cuts.cutMask & cutMask);
}

int OffEgSel::getCutCode(const OffPho& pho,int cutMask)const
{
  if(fabs(pho.detEta())<1.5) return getCutCode(pho,ebCutValues_,cutMask);
  else return getCutCode(pho,eeCutValues_,cutMask);
}

//photons automatically fail any track cut
int OffEgSel::getCutCode(const OffPho& pho,const EgCutValues& cuts,int cutMask)
{ 
  int cutCode = 0x0;
  //kinematic cuts
  if(pho.et()< cuts.minEt) cutCode |= EgCutCodes::ET;
  if(fabs(pho.etaSC())< cuts.minEta || fabs(pho.etaSC())>cuts.maxEta) cutCode |= EgCutCodes::DETETA;
  if(pho.isGap()) cutCode |= EgCutCodes::CRACK;
  //track cuts (all fail)
  cutCode |=EgCutCodes::DETAIN;
  cutCode |=EgCutCodes::DPHIIN;
  cutCode |=EgCutCodes::INVEINVP;
  //supercluster cuts
  if(pho.hOverE()> cuts.maxHadem && pho.hOverE()*pho.energy()>cuts.maxHadEnergy) cutCode |= EgCutCodes::HADEM;
  if(pho.sigmaIEtaIEta()>cuts.maxSigmaIEtaIEta) cutCode |= EgCutCodes::SIGMAIETAIETA; 
  if(pho.sigmaEtaEta()>cuts.maxSigmaEtaEta) cutCode |= EgCutCodes::SIGMAETAETA; 
  //----Morse--------
  //if(pho.r9()<cuts.minR9) cutCode |= EgCutCodes::R9;
  if(pho.r9()<cuts.minR9) cutCode |= EgCutCodes::MINR9;
  if(pho.r9()>cuts.maxR9) cutCode |= EgCutCodes::MAXR9;
  //----------------
  //std isolation cuts
  if(pho.isolEm()>( cuts.isolEmConstTerm + cuts.isolEmGradTerm*(pho.et()<cuts.isolEmGradStart ? 0. : (pho.et()-cuts.isolEmGradStart)))) cutCode |=EgCutCodes::ISOLEM;
  if(pho.isolHad()> (cuts.isolHadConstTerm + cuts.isolHadGradTerm*(pho.et()<cuts.isolHadGradStart ? 0. : (pho.et()-cuts.isolHadGradStart)))) cutCode |=EgCutCodes::ISOLHAD;
  if(pho.isolPtTrks() > (cuts.isolPtTrksConstTerm + cuts.isolPtTrksGradTerm*(pho.et()<cuts.isolPtTrksGradStart ? 0. : (pho.et()-cuts.isolPtTrksGradStart))))cutCode |=EgCutCodes::ISOLPTTRKS;
  if(pho.isolNrTrks() > cuts.isolNrTrksConstTerm) cutCode |=EgCutCodes::ISOLNRTRKS;

  //hlt isolation cuts
  cutCode |=EgCutCodes::HLTISOLTRKSELE; //automatically fails ele track isolation
  if(pho.et()<=0.){ //even it if et<=0, we give it a shot at passing isolation. Note this should be an impossible case
    if(pho.hltIsolTrks() > cuts.maxHLTIsolTrksPho) cutCode |=EgCutCodes::HLTISOLTRKSPHO;
    if(pho.hltIsolHad() > cuts.maxHLTIsolHad) cutCode |=EgCutCodes::HLTISOLHAD;
    if(pho.hltIsolEm() > cuts.maxHLTIsolEm) cutCode |=EgCutCodes::HLTISOLEM;
  }else{ 
    if(pho.hltIsolTrks() > cuts.maxHLTIsolTrksPho && pho.hltIsolTrks()/pho.et() > cuts.maxHLTIsolTrksPhoOverPt &&
       pho.hltIsolTrks()/pho.et()/pho.et() > cuts.maxHLTIsolTrksPhoOverPt2 ) cutCode |=EgCutCodes::HLTISOLTRKSPHO;
    if(pho.hltIsolHad() > cuts.maxHLTIsolHad && pho.hltIsolHad()/pho.et() > cuts.maxHLTIsolHadOverEt &&
       pho.hltIsolHad()/pho.et()/pho.et() > cuts.maxHLTIsolHadOverEt2) cutCode |=EgCutCodes::HLTISOLHAD;
    if(pho.hltIsolEm() > cuts.maxHLTIsolEm && pho.hltIsolEm()/pho.et() > cuts.maxHLTIsolEmOverEt &&
       pho.hltIsolEm()/pho.et()/pho.et() > cuts.maxHLTIsolEmOverEt2) cutCode |=EgCutCodes::HLTISOLEM;
  }

  //track cuts, photon will automatically fail them (for now)
  cutCode |=EgCutCodes::CTFTRACK;
  cutCode |=EgCutCodes::HLTDETAIN;
  cutCode |=EgCutCodes::HLTDPHIIN;
  cutCode |=EgCutCodes::HLTINVEINVP;
  
  return (cutCode & cuts.cutMask & cutMask) ;
}
