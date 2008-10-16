#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"
#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"

VariableHelper * VariableHelperInstance::SetVariableHelperUniqueInstance_=0;
std::map<std::string, VariableHelper* > VariableHelperInstance::multipleInstance_ = std::map<std::string, VariableHelper* >();

VariableHelper::VariableHelper(const edm::ParameterSet & iConfig){
  std::vector<std::string> psetNames;
  iConfig.getParameterSetNames(psetNames);
  for (uint i=0;i!=psetNames.size();++i){
    std::string & vname=psetNames[i];
    edm::ParameterSet vPset=iConfig.getParameter<edm::ParameterSet>(psetNames[i]);
    //std::string type=vPset.getParameter<std::string>("type");
    std::string type="helper";
    if (type=="helper"){
      std::string method=vPset.getParameter<std::string>("method");
      variables_[vname]=CachingVariableFactory::get()->create(method,vname,vPset);
    }
    else{
      //type not recognized
      throw;
    }
  }
}

void VariableHelper::setHolder(std::string hn){
  std::map<std::string, CachingVariable*> ::const_iterator it = variables_.begin();
  std::map<std::string, CachingVariable*> ::const_iterator it_end = variables_.end();
  for (;it!=it_end;++it)  it->second->setHolder(hn);
}

/*
void VariableHelper::update(const edm::Event & e, const edm::EventSetup & es) const
{
  ev_=&e;
  es_=&es;
}
*/

const CachingVariable* VariableHelper::variable(std::string name) const{ 
  std::map<std::string, CachingVariable*> ::const_iterator v=variables_.find(name);
  if (v!=variables_.end())
    return v->second;
  else
    {
      edm::LogError("VariableHelper")<<"I don't know anything named: "<<name;
      return 0;
    }
}


/*
double VariableHelper::operator() (std::string & name,const edm::Event & iEvent) const{  
  const CachingVariable* v = variable(name);
  return (*v)();
}

double VariableHelper::operator() (std::string name) const{
  const CachingVariable* v = variable(name);
  return (*v)();
}
*/
