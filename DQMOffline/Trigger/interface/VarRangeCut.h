#ifndef DQMOffline_Trigger_VarRangeCut_h
#define DQMOffline_Trigger_VarRangeCut_h


//********************************************************************************
//
// Description:
//   A object containing a minimal set of selection cuts. 
//   These selection cuts are intended to be simple selections on kinematic variables.
//   Currently these are implimented as simple allowed ranges  X<var<Y which are ORed together
//   So for example we may want to have an eta cut of 0<|eta|<1.4442 || 1.556<|eta|<2.5
//   
// Implimentation:
//   std::function holds the function which generates the variable from the object
//   the name of the variable is also stored so we can determine if we should not apply
//   a given selection cut
//   
// Author : Sam Harper , RAL, May 2017
//
//***********************************************************************************

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DQMOffline/Trigger/interface/FunctionDefs.h"

#include <boost/algorithm/string.hpp>

template<typename ObjType>
class VarRangeCut {
public:
  explicit VarRangeCut(const edm::ParameterSet& config);
  static edm::ParameterSetDescription makePSetDescription();
  
  bool operator()(const ObjType& obj)const;
  const std::string& varName()const{return varName_;}

private:
  std::string varName_;
  std::function<float(const ObjType&)> varFunc_;
  std::vector<std::pair<float,float> > allowedRanges_;
};

template<typename ObjType>
VarRangeCut<ObjType>::VarRangeCut(const edm::ParameterSet& config)
{
  varName_ = config.getParameter<std::string>("rangeVar");
  varFunc_ = hltdqm::getUnaryFuncFloat<ObjType>(varName_);
  auto ranges = config.getParameter<std::vector<std::string> >("allowedRanges");
  for(auto range: ranges){
    std::vector<std::string> splitRange;
    boost::split(splitRange,range,boost::is_any_of(":"));
    if(splitRange.size()!=2) throw cms::Exception("ConfigError") <<"in VarRangeCut::VarRangeCut range "<<range<<" is not of format X:Y"<<std::endl;
    allowedRanges_.push_back({std::stof(splitRange[0]),std::stof(splitRange[1])});
  }
}

template<typename ObjType>
edm::ParameterSetDescription VarRangeCut<ObjType>::makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>("rangeVar","");
  desc.add<std::vector<std::string> >("allowedRanges",std::vector<std::string>());
  return desc;
}

template<typename ObjType>
bool VarRangeCut<ObjType>::operator()(const ObjType& obj)const
{
  if(!varFunc_) return true; //auto pass if we dont specify a variable function
  else{ 
    float varVal = varFunc_(obj);
    for(auto& range : allowedRanges_){
      if(varVal>=range.first && varVal<range.second) return true;
    }
    return false;
  }
}

#endif
