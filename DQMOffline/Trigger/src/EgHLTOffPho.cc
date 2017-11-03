#include <cmath>

#include "DQMOffline/Trigger/interface/EgHLTOffPho.h"

using namespace egHLT;

float OffPho::sigmaEtaEta()const
{
  if(std::fabs(etaSC())<1.479) return clusShapeData_.sigmaEtaEta; //barrel case, no correction
  else{ //endcap, need to apply eta correction
    return clusShapeData_.sigmaEtaEta - 0.02*( std::fabs(etaSC()) - 2.3);
  } 

}

int OffPho::trigCutsCutCode(const TrigCodes::TrigBitSet& trigger)const
{
  //yes maybe a sorted vector might be better but 1) its small and 2) bitset doesnt support < operator
  //okay laugh, for some reason I cant overload the == operator (brain just not working), hence the non stl'y way
  //std::vector<std::pair<TrigCodes::TrigBitSet,int> >::const_iterator it;
  //it = std::find(trigCutsCodes_.begin(),trigCutsCodes_.end(),trigger);
  //if(it!=trigCutsCodes_.end()) return it->second;
  //else return 0; //defaults to passing

  for(auto const & trigCutsCutCode : trigCutsCutCodes_) if(trigger==trigCutsCutCode.first) return trigCutsCutCode.second;
  return 0; //defaults to passing
}
