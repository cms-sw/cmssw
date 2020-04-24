#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"
#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"

#include <iomanip>

VariableHelper::VariableHelper(const edm::ParameterSet & iConfig, edm::ConsumesCollector& iC){
  std::vector<std::string> psetNames;
  iConfig.getParameterSetNames(psetNames);
  for (unsigned int i=0;i!=psetNames.size();++i){
    std::string & vname=psetNames[i];
    edm::ParameterSet vPset=iConfig.getParameter<edm::ParameterSet>(psetNames[i]);
    std::string method=vPset.getParameter<std::string>("method");

    CachingVariableFactory::get()->create(method,CachingVariable::CachingVariableFactoryArg(vname,variables_,vPset), iC);
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

std::string VariableHelper::printValues(const edm::Event & event) const{
  std::stringstream ss;
  iterator it = variables_.begin();
  iterator it_end = variables_.end();
  ss<<std::setw(10)<<event.id().run()<<" : "
    <<std::setw(10)<<event.id().event();
  for (;it!=it_end;++it) {
    if (it->second->compute(event))
      ss<<" : "<<it->first<<"="<<(*it->second)(event);
    else
      ss<<" : "<<it->first<<" N/A";
  }
  return ss.str();
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
