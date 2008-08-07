#include "DQMOffline/Trigger/interface/EgHLTOffEleSel.h"

#include "DQMOffline/Trigger/interface/CutCodes.h"
#include "DQMOffline/Trigger/interface/EleTypeCodes.h"

EgHLTOffEleSel::EgHLTOffEleSel()
{

}


int EgHLTOffEleSel::getCutCode(const EgHLTOffEle& ele,int cutMask)const
{
  const CutValues& cuts = getCuts(ele.classification());
  if(&cuts!=NULL) return getCutCode(ele,cuts,cutMask);
  else return ~0x0;
}


const CutValues& EgHLTOffEleSel::getCuts(int type)const
{
  int cutIndex=0;
  int nrMatches = 0;
  for(unsigned index=0;index<cutValues_.size();index++){
    //std::cout <<"valid ele type code "<<std::hex<<cutValues_[index].validEleTypes<<" ele type code "<< EleTypeCodes::makeTypeCode(type) <<" should have code "<<EleTypeCodes::getCode("barrel:golden:narrow:bigBrem:showering")<<std::dec<<" ele type "<<type<<std::endl;
    if( (EleTypeCodes::makeTypeCode(type) & cutValues_[index].validEleTypes) == EleTypeCodes::makeTypeCode(type)){
      if(nrMatches==0) cutIndex = index;
      nrMatches++;
    }
  }
  if(nrMatches>=1){
    if(nrMatches>1) std::cout <<"EgHLTOffEleSel::getCuts: Warning have "<<nrMatches<<" for electron type "<<type<<std::endl;
    return cutValues_[cutIndex];
  }else{
    //std::cout <<"EgHLTOffEleSel::getCuts: Warning have "<<nrMatches<<" for electron type "<<type<<std::endl;
    CutValues* nullPointer=0; //there has to be a better way, basically I'm in a dire need of a way to return a null object safely when I dont have it and I'm not using exceptions
    return *nullPointer;
  }
}

CutValues& EgHLTOffEleSel::getCuts(int type)
{
  const EgHLTOffEleSel* constThis = this;
  const CutValues& constCuts = constThis->getCuts(type);
  CutValues& cuts = *(const_cast<CutValues*>(&constCuts)); //is this evil code? Possibly, but I think its safe as all I'm doing is basically code reuse by using the const function and then casting the returned value to non const as in this case the object is really non-const 
  return cuts;

}

int EgHLTOffEleSel::getCutCode(const EgHLTOffEle& ele,const CutValues& cuts,int cutMask)
{ 
  int cutCode = 0x0;
  if(ele.et()< cuts.minEtCut) cutCode |= CutCodes::ET;
  if(fabs(ele.etaSC())< cuts.minEtaCut || fabs(ele.etaSC())>cuts.maxEtaCut) cutCode |= CutCodes::DETETA;
  if(ele.classification()==40 && cuts.rejectCracks) cutCode |= CutCodes::CRACK;
  if((ele.epIn()< cuts.minEpInCut || ele.epIn()> cuts.maxEpInCut) &&  ele.etSC()<cuts.epInReleaseEtCut ) cutCode |=CutCodes::EPIN;
  if(fabs(ele.dEtaIn()) > cuts.maxDEtaInCut ) cutCode |=CutCodes::DETAIN;
  if(fabs(ele.dPhiIn()) > cuts.maxDPhiInCut ) cutCode |=CutCodes::DPHIIN;
  if(ele.hOverE()> cuts.maxHademCut) cutCode |= CutCodes::HADEM;
  if(ele.epOut()< cuts.minEpOutCut || ele.epOut()>cuts.maxEpOutCut) cutCode |=CutCodes::EPOUT;
  if(fabs(ele.dPhiOut()) > cuts.maxDPhiOutCut) cutCode |=CutCodes::DPHIOUT;
  if(ele.invEOverInvP() > cuts.maxInvEInvPCut) cutCode |= CutCodes::INVEINVP;
  if(ele.bremFrac() < cuts.minBremFracCut) cutCode |= CutCodes::BREMFRAC;
  //if(ele.e9OverE25() < cuts.minE9E25Cut) cutCode |= CutCodes::E9OVERE25;
  if(ele.sigmaEtaEta()<cuts.minSigmaEtaEtaCut || ele.sigmaEtaEta()>cuts.maxSigmaEtaEtaCut) cutCode |= CutCodes::SIGMAETAETA;
  //if(ele.sigmaPhiPhi()<cuts.minSigmaPhiPhiCut || ele.sigmaPhiPhi()>cuts.maxSigmaPhiPhiCut) cutCode |= CutCodes::SIGMAPHIPHI;
  if(ele.isolEm()>( cuts.minIsolEmConstCut + cuts.isolEmGradCut*ele.et())) cutCode |=CutCodes::ISOLEM;
  if(ele.isolHad()> (cuts.minIsolHadConstCut + cuts.isolHadGradCut*ele.et())) cutCode |=CutCodes::ISOLHAD;
  if(ele.isolPtTrks() > (cuts.minIsolPtTrksConstCut + cuts.isolPtTrksGradCut*ele.et())) cutCode |=CutCodes::ISOLPTTRKS;
  if(ele.isolNrTrks() > cuts.minIsolNrTrksConstCut) cutCode |=CutCodes::ISOLNRTRKS;

  return (cutCode & cuts.cutMask & cutMask) ;
}


void EgHLTOffEleSel::setHighNrgy()
{
  clearCuts();
 
  CutValues ebCuts;
  CutValues eeCuts;
  ebCuts.setEBHighNrgy(CutCodes::getCode("et:detEta:crack:dEtaIn:dPhiIn:hadem:sigmaEtaEta:isolEm:isolHad:isolPtTrks"));
  ebCuts.validEleTypes = EleTypeCodes::getCode("barrel:golden:narrow:bigBrem:showering:crack");
  eeCuts.setEEHighNrgy(CutCodes::getCode("et:detEta:crack:dEtaIn:dPhiIn:hadem:sigmaEtaEta:isolEm:isolHad:isolPtTrks"));
  eeCuts.validEleTypes = EleTypeCodes::getCode("endcap:golden:narrow:bigBrem:showering:crack");
  addCuts(ebCuts);
  addCuts(eeCuts);
}  


void EgHLTOffEleSel::setPreSel()
{
  clearCuts();
 
  CutValues ebCuts;
  CutValues eeCuts;
  ebCuts.setEBPreSel(CutCodes::getCode("et:detEta:crack:dEtaIn:dPhiIn:hadem"));
  ebCuts.validEleTypes = EleTypeCodes::getCode("barrel:golden:narrow:bigBrem:showering:crack");
  eeCuts.setEEPreSel(CutCodes::getCode("et:detEta:crack:dEtaIn:dPhiIn:hadem"));
  eeCuts.validEleTypes = EleTypeCodes::getCode("endcap:golden:narrow:bigBrem:showering:crack");
  addCuts(ebCuts);
  addCuts(eeCuts);
}  

void EgHLTOffEleSel::setPreSelWithEp()
{
  clearCuts();
 
  CutValues ebCuts;
  CutValues eeCuts;
  ebCuts.setEBPreSel(CutCodes::getCode("et:detEta:crack:dEtaIn:dPhiIn:hadem:epIn"));
  ebCuts.validEleTypes = EleTypeCodes::getCode("barrel:golden:narrow:bigBrem:showering:crack");
  eeCuts.setEEPreSel(CutCodes::getCode("et:detEta:crack:dEtaIn:dPhiIn:hadem:epIn"));
  eeCuts.validEleTypes = EleTypeCodes::getCode("endcap:golden:narrow:bigBrem:showering:crack");
  addCuts(ebCuts);
  addCuts(eeCuts);
}

void EgHLTOffEleSel::setCutMask(int cutMask,int eleType)
{
  for(size_t cutValNr=0;cutValNr<cutValues_.size();cutValNr++){
    if((eleType&cutValues_[cutValNr].validEleTypes)!=0){
      cutValues_[cutValNr].cutMask = cutMask;
    }
  }
   
}

void EgHLTOffEleSel::removeCuts(int cutCode,int eleType)
{
  for(size_t cutValNr=0;cutValNr<cutValues_.size();cutValNr++){
    if((eleType&cutValues_[cutValNr].validEleTypes)!=0){
      cutValues_[cutValNr].cutMask &= ~cutCode;
    }
  }
}

void EgHLTOffEleSel::setMinEt(float minEt,int eleType)
{
  for(size_t cutValNr=0;cutValNr<cutValues_.size();cutValNr++){
    if((eleType&cutValues_[cutValNr].validEleTypes)!=0){
      cutValues_[cutValNr].minEtCut = minEt;
    }
  }
}
  
