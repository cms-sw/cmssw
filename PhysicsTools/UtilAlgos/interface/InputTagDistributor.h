#ifndef InputTagDistributor_H
#define InputTagDistributor_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>
#include <iostream>

class InputTagDistributor{
 private:
  static InputTagDistributor* SetInputTagDistributorUniqueInstance_;
  static std::map<std::string, InputTagDistributor*> multipleInstance_;
  
 public:
  InputTagDistributor(const edm::ParameterSet & pset){
    std::vector< std::string > inpuTags = pset.getParameterNamesForType<edm::InputTag>();
    for (std::vector< std::string >::iterator i=inpuTags.begin();i!=inpuTags.end();++i)
      inputTags_[*i]=pset.getParameter<edm::InputTag>(*i);
  }
  const edm::InputTag & inputTag(std::string s){ 
    std::map<std::string, edm::InputTag>::iterator findMe = inputTags_.find(s);
    if (findMe!=inputTags_.end())
      return findMe->second;
    else{
      std::stringstream known;
      for (findMe=inputTags_.begin();findMe!=inputTags_.end();++findMe)
	known<<findMe->first<<" ---> "<<findMe->second<<"\n";
      edm::LogError("InputTagDistributor")<<" cannot distribute InputTag: "<<s<<"\n knonw mapping is:\n"<<known.str();
    }
    return inputTags_[s];
  }

  static InputTagDistributor & init(std::string user, const edm::ParameterSet & iConfig){
    if (multipleInstance_.find(user)!=multipleInstance_.end()){
      std::cerr<<user<<" InputTagDistributor user already defined."<<std::endl;
      throw;}
    else SetInputTagDistributorUniqueInstance_ = new InputTagDistributor(iConfig);
    multipleInstance_[user] = SetInputTagDistributorUniqueInstance_;
    return (*SetInputTagDistributorUniqueInstance_);
  }
  static InputTagDistributor & set(std::string & user){
    if (multipleInstance_.find(user)==multipleInstance_.end()){
      std::cerr<<user<<" InputTagDistributor  user not defined. but it does not matter."<<std::endl;
      //      throw;}
    }
    else return (*SetInputTagDistributorUniqueInstance_);
  }
  static InputTagDistributor & get(){
    if (!SetInputTagDistributorUniqueInstance_){
      std::cerr<<" SetInputTagDistributorUniqueInstance_ is not valid."<<std::endl;
      throw;
    }
    else{ return (*SetInputTagDistributorUniqueInstance_);}
  }
  
  static edm::InputTag retrieve(std::string src,const edm::ParameterSet & pset){
    //if used without setting any InputTag mapping
    if (multipleInstance_.size()==0)
      return pset.getParameter<edm::InputTag>(src);
    
    // some mapping was setup
    InputTagDistributor & which=get();
    std::map<std::string, InputTagDistributor*>::iterator inverseMap=multipleInstance_.begin();
    std::map<std::string, InputTagDistributor*>::iterator inverseMap_end=multipleInstance_.end();
    for (;inverseMap!=inverseMap_end;++inverseMap) if (inverseMap->second==&which) break;
    LogDebug("InputTagDistributor")<<"typeCode: "<<pset.retrieve(src).typeCode()
	     <<"\n"<<pset.dump()<<"\n"
	     <<"looking for: "<<src
	     <<" by user: "<< inverseMap->first
	     <<std::endl;
    std::string typeCode;
    typeCode+=pset.retrieve(src).typeCode();
    std::string iTC;iTC+='S';
    if (typeCode==iTC)
      return which.inputTag(pset.getParameter<std::string>(src));
    else
      return pset.getParameter<edm::InputTag>(src);
  }

 private:
  std::map<std::string, edm::InputTag> inputTags_;
};

#endif
