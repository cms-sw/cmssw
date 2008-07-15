#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"

float EgHLTOffEle::sigmaEtaEta()const
{
  if(clusShape_!=NULL){
    if(fabs(etaSC())<1.479) return sqrt(clusShape_->covEtaEta()); //barrel case, no correction
    else{ //endcap, need to apply eta correction
      float unCorrSigmaEtaEta = sqrt(clusShape_->covEtaEta());
      return unCorrSigmaEtaEta - 0.02*( fabs(etaSC()) - 2.3);
    }
  }else return 999.;

}
