#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"
#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"


VariableHelper::VariableHelper(const edm::ParameterSet & iConfig){
  std::vector<std::string> psetNames;
  iConfig.getParameterSetNames(psetNames);
  for (uint i=0;i!=psetNames.size();++i){
    std::string & vname=psetNames[i];
    edm::ParameterSet vPset=iConfig.getParameter<edm::ParameterSet>(psetNames[i]);
    std::string method=vPset.getParameter<std::string>("method");

    CachingVariableFactory::get()->create(method,CachingVariable::CachingVariableFactoryArg(vname,variables_,vPset));
  }

}

void VariableHelper::setHolder(std::string hn){
  iterator it = variables_.begin();
  iterator it_end = variables_.end();
  for (;it!=it_end;++it)  it->second->setHolder(hn);
}

void VariableHelper::print() const{
  iterator it = variables_.begin();
  iterator it_end = variables_.end();
  for (;it!=it_end;++it)  it->second->print();
}

const CachingVariable* VariableHelper::variable(std::string name) const{ 
  iterator v=variables_.find(name);
  if (v!=variables_.end())
    return v->second;
  else
    {
      edm::LogError("VariableHelper")<<"I don't know anything named: "<<name
				     <<" list of available variables follows.";
      print();
      return 0;
    }
}
