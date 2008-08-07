#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"

float EgHLTOffEle::sigmaEtaEta()const
{
  if(fabs(etaSC())<1.479) return sigmaEtaEta_; //barrel case, no correction
  else{ //endcap, need to apply eta correction
    return sigmaEtaEta_ - 0.02*( fabs(etaSC()) - 2.3);
  } 

}
