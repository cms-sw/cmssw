#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"

using namespace egHLT;

float OffEle::sigmaEtaEta()const
{
  if(fabs(etaSC())<1.479) return clusShapeData_.sigmaEtaEta; //barrel case, no correction
  else{ //endcap, need to apply eta correction
    return clusShapeData_.sigmaEtaEta - 0.02*( fabs(etaSC()) - 2.3);
  } 

}

//defining the == operator
//bool operator==(const std::pair<TrigCodes::TrigBitSet,int>& lhs,const TrigCodes::TrigBitSet& rhs){return lhs.first==rhs;}
//bool operator==(const TrigCodes::TrigBitSet& lhs,const std::pair<TrigCodes::TrigBitSet,int>& rhs){return lhs==rhs.first;}

int OffEle::trigCutsCutCode(const TrigCodes::TrigBitSet& trigger)const
{
  //yes maybe a sorted vector might be better but 1) its small and 2) bitset doesnt support < operator
  //okay laugh, for some reason I cant overload the == operator (brain just not working), hence the non stl'y way
  //std::vector<std::pair<TrigCodes::TrigBitSet,int> >::const_iterator it;
  //it = std::find(trigCutsCodes_.begin(),trigCutsCodes_.end(),trigger);
  //if(it!=trigCutsCodes_.end()) return it->second;
  //else return 0; //defaults to passing

  for(size_t i=0;i<trigCutsCutCodes_.size();i++) if(trigger==trigCutsCutCodes_[i].first) return trigCutsCutCodes_[i].second;
  return 0; //defaults to passing
}


// float EgHLTOffEle::sigmaIEtaIEta()const
// {
//   if(fabs(etaSC())<1.479) return clusShapeData_.sigmaIEtaIEta; //barrel case, no correction
//   else{ //endcap, need to apply eta correction
//     return clusShapeData_.sigmaIEtaIEta - 0.02*( fabs(etaSC()) - 2.3);
//   } 

// }
