#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"

float EgHLTOffEle::sigmaEtaEta()const
{
  if(fabs(etaSC())<1.479) return clusShapeData_.sigmaEtaEta; //barrel case, no correction
  else{ //endcap, need to apply eta correction
    return clusShapeData_.sigmaEtaEta - 0.02*( fabs(etaSC()) - 2.3);
  } 

}
// float EgHLTOffEle::sigmaIEtaIEta()const
// {
//   if(fabs(etaSC())<1.479) return clusShapeData_.sigmaIEtaIEta; //barrel case, no correction
//   else{ //endcap, need to apply eta correction
//     return clusShapeData_.sigmaIEtaIEta - 0.02*( fabs(etaSC()) - 2.3);
//   } 

// }
