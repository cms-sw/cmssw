#ifndef ConfigurableAnalysis_VariableHelper_H
#define ConfigurableAnalysis_VariableHelper_H

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class VariableHelper {
 public:
  VariableHelper(const edm::ParameterSet & iConfig);  
  typedef std::map<std::string, CachingVariable*>::const_iterator iterator;

  const CachingVariable* variable(std::string name)const ;

  iterator begin() { return variables_.begin();}
  iterator end() { return variables_.end();}

  void setHolder(std::string hn);
 private:
  std::map<std::string, CachingVariable*> variables_;
};




class VariableHelperInstance {
 private:
  static VariableHelper * SetVariableHelperUniqueInstance_;
  static std::map<std::string, VariableHelper* > multipleInstance_;

 public:

  static VariableHelper & init(std::string user, const edm::ParameterSet & iConfig){
    if (multipleInstance_.find(user)!=multipleInstance_.end()){
      std::cerr<<user<<" VariableHelper user already defined."<<std::endl;
      throw;}
    else SetVariableHelperUniqueInstance_ = new VariableHelper(iConfig);
    multipleInstance_[user] = SetVariableHelperUniqueInstance_;
    SetVariableHelperUniqueInstance_->setHolder(user);
    return (*SetVariableHelperUniqueInstance_);
  }
  
  static VariableHelper & get(){
    if (!SetVariableHelperUniqueInstance_)
      {
	std::cerr<<" none of VariableHelperUniqueInstance_ or SetVariableHelperUniqueInstance_ is valid."<<std::endl;
	throw;
      }
    else return (*SetVariableHelperUniqueInstance_);
  }

  static VariableHelper & set(std::string user){
    std::map<std::string, VariableHelper* >::iterator f=multipleInstance_.find(user);
    if (f == multipleInstance_.end()){
      std::cerr<<user<<" VariableHelper user not defined."<<std::endl;
      throw;
    }
    else{
      SetVariableHelperUniqueInstance_ = (f->second);
      return (*SetVariableHelperUniqueInstance_);
    }
  }
};

#endif
