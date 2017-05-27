#ifndef DQMOffline_Trigger_VarRangeCutColl_h
#define DQMOffline_Trigger_VarRangeCutColl_h

//********************************************************************************
//
// Description:
//   A collection cut minimal selection cuts.
//   These selection cuts are intended to be simple selections on kinematic variables.
//   The cuts are ANDed together
//   intended use case cuts on et and eta
//   However individual cuts can be skipped, so you can disable say only the Et cut
//   when  you're doing a turn on
//   
// Implimentation:
//   Basically a vector of VarRangeCuts which a nice operator() function to make it 
//   easy to use 
//   
// Author : Sam Harper , RAL, May 2017
//
//***********************************************************************************

#include "DQMOffline/Trigger/interface/VarRangeCut.h"

template<typename ObjType>
class VarRangeCutColl {
public:
  explicit VarRangeCutColl(const std::vector<edm::ParameterSet>& configs){
    for(const auto & cutConfig : configs) rangeCuts_.emplace_back(VarRangeCut<ObjType>(cutConfig));
  }
  
  //if no cuts are defined, it returns true
  bool operator()(const ObjType& obj)const{
    for(auto& cut : rangeCuts_){
      if(!cut(obj)) return false;
    }
    return true;
  }

  //this version allows us to skip a range check for a specificed variable
  //okay this feature requirement was missed in the initial (very rushed) design phase
  //and thats why its now hacked in 
  //basically if you're applying an Et cut, you want to automatically turn it of
  //when you're making a turn on curve...
  //if no cuts are defined, it returns true
  bool operator()(const ObjType& obj,const std::string& varToSkip)const{
    for(auto& cut : rangeCuts_){
      if(cut.varName()==varToSkip) continue;
      if(!cut(obj)) return false;
    }
    return true;
  }
  //for multiple cuts to skip
  bool operator()(const ObjType& obj,const std::vector<std::string>& varsToSkip)const{
    for(auto& cut : rangeCuts_){
      if(std::find(varsToSkip.begin(),varsToSkip.end(),cut.varName())!=varsToSkip.end()) continue;
      if(!cut(obj)) return false;
    }
    return true;
  }
private:
  std::vector<VarRangeCut<ObjType> > rangeCuts_;
};

#endif
