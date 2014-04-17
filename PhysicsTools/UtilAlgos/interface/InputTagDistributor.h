#ifndef InputTagDistributor_H
#define InputTagDistributor_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include <map>
#include <iostream>

class InputTagDistributor{
 public:
  InputTagDistributor(const edm::ParameterSet & pset, edm::ConsumesCollector& iC){
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

 private:
  std::map<std::string, edm::InputTag> inputTags_;
};


class InputTagDistributorService{
 private:
  InputTagDistributor* SetInputTagDistributorUniqueInstance_;
  std::map<std::string, InputTagDistributor*> multipleInstance_;

 public:
  InputTagDistributorService(const edm::ParameterSet & iConfig,edm::ActivityRegistry & r ){
    r.watchPreModule(this, &InputTagDistributorService::preModule );
  };
  ~InputTagDistributorService(){};

  InputTagDistributor & init(std::string user, const edm::ParameterSet & iConfig, edm::ConsumesCollector&& iC){
    if (multipleInstance_.find(user)!=multipleInstance_.end()){
      std::cerr<<user<<" InputTagDistributor user already defined."<<std::endl;
      throw;}
    else SetInputTagDistributorUniqueInstance_ = new InputTagDistributor(iConfig, iC);
    multipleInstance_[user] = SetInputTagDistributorUniqueInstance_;
    return (*SetInputTagDistributorUniqueInstance_);
  }
  void preModule(const edm::ModuleDescription& desc){
    //does a set with the module name, except that it does not throw on non-configured modules
    std::map<std::string, InputTagDistributor*>::iterator f=multipleInstance_.find(desc.moduleLabel());
    if (f != multipleInstance_.end()) SetInputTagDistributorUniqueInstance_ = f->second;
    else{
      //do not say anything but set it to zero to get a safe crash in get() if ever called
      SetInputTagDistributorUniqueInstance_=0;}
  }
  /*  InputTagDistributor & set(std::string & user){
    std::map<std::string, InputTagDistributor*>::iterator f=multipleInstance_.find(user);
    if (f == multipleInstance_.end()){
      std::cerr<<user<<" InputTagDistributor  user not defined. but it does not matter."<<std::endl;
      //      throw;}
    }
    else {
      SetInputTagDistributorUniqueInstance_ = f->second;
      return (*SetInputTagDistributorUniqueInstance_);
    }
    }*/
  InputTagDistributor & get(){
    if (!SetInputTagDistributorUniqueInstance_){
      std::cerr<<" SetInputTagDistributorUniqueInstance_ is not valid."<<std::endl;
      throw;
    }
    else{ return (*SetInputTagDistributorUniqueInstance_);}
  }

  edm::InputTag retrieve(std::string src,const edm::ParameterSet & pset){
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
};



#endif
