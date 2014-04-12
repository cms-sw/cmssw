#ifndef ConfigurableAnalysis_VariableHelper_H
#define ConfigurableAnalysis_VariableHelper_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

class VariableHelper {
 public:
  VariableHelper(const edm::ParameterSet & iConfig, edm::ConsumesCollector& iC);
  ~VariableHelper() {
    for (iterator it = variables_.begin() ; it!=variables_.end() ;++it){
      delete it->second;
    }
  }
  typedef std::map<std::string,const CachingVariable*>::const_iterator iterator;

  const CachingVariable* variable(std::string name)const ;

  iterator begin() { return variables_.begin();}
  iterator end() { return variables_.end();}

  void setHolder(std::string hn);
  void print() const;
  std::string printValues(const edm::Event & event) const;
 private:
  std::map<std::string,const CachingVariable*> variables_;
};




class VariableHelperService {
 private:
  VariableHelper * SetVariableHelperUniqueInstance_;
  std::map<std::string, VariableHelper* > multipleInstance_;

  bool printValuesForEachEvent_;
  std::string printValuesForEachEventCategory_;
 public:
  VariableHelperService(const edm::ParameterSet & iConfig,edm::ActivityRegistry & r ){
    r.watchPreModule(this, &VariableHelperService::preModule );
    r.watchPostProcessEvent(this, &VariableHelperService::postProcess );
    printValuesForEachEvent_ = iConfig.exists("printValuesForEachEventCategory");
    if (printValuesForEachEvent_)
      printValuesForEachEventCategory_ = iConfig.getParameter<std::string>("printValuesForEachEventCategory");
  }
  ~VariableHelperService(){
    for (std::map<std::string, VariableHelper* > :: iterator it=multipleInstance_.begin(); it!=multipleInstance_.end(); ++it){
      delete it->second;
    }
  }

  VariableHelper & init(std::string user, const edm::ParameterSet & iConfig, edm::ConsumesCollector&& iC){
    if (multipleInstance_.find(user)!=multipleInstance_.end()){
      std::cerr<<user<<" VariableHelper user already defined."<<std::endl;
      throw;}
    else SetVariableHelperUniqueInstance_ = new VariableHelper(iConfig, iC);
    multipleInstance_[user] = SetVariableHelperUniqueInstance_;
    SetVariableHelperUniqueInstance_->setHolder(user);

    SetVariableHelperUniqueInstance_->print();
    return (*SetVariableHelperUniqueInstance_);
  }

  VariableHelper & get(){
    if (!SetVariableHelperUniqueInstance_)
      {
	std::cerr<<" none of VariableHelperUniqueInstance_ or SetVariableHelperUniqueInstance_ is valid."<<std::endl;
	throw;
      }
    else return (*SetVariableHelperUniqueInstance_);
  }

  void preModule(const edm::ModuleDescription& desc){
    //does a set with the module name, except that it does not throw on non-configured modules
    std::map<std::string, VariableHelper* >::iterator f=multipleInstance_.find(desc.moduleLabel());
    if (f != multipleInstance_.end())  SetVariableHelperUniqueInstance_ = (f->second);
    else {
      //do not say anything but set it to zero to get a safe crash in get() if ever called
      SetVariableHelperUniqueInstance_ =0;}
  }

  void postProcess(const edm::Event & event, const edm::EventSetup & setup){
    if (!printValuesForEachEvent_) return;
    std::map<std::string, VariableHelper* >::iterator f= multipleInstance_.begin();
    for (; f!=multipleInstance_.end();++f){
      //      std::cout<<" category is: "<<printValuesForEachEventCategory_+"|"+f->first<<std::endl;
      //      std::cout<<f->first<<"\n"	       <<f->second->printValues(event);

      edm::LogInfo(printValuesForEachEventCategory_+"|"+f->first)<<f->first<<"\n"
								 <<f->second->printValues(event);
    }
  }

  VariableHelper & set(std::string user){
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
