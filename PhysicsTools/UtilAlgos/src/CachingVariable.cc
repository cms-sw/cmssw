#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"
#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"



CachingVariable::evalType VarSplitter::eval(const edm::Event & iEvent) const{
  const CachingVariable * var=edm::Service<VariableHelperService>()->get().variable(var_);
  if (!var->compute(iEvent)) return std::make_pair(false,0);

  double v=(*var)(iEvent);
  if (v<slots_.front()){
    if (useUnderFlow_) return std::make_pair(true,0);
    else return std::make_pair(false,0);
  }
  if (v>=slots_.back()){
    if (useOverFlow_) return std::make_pair(true,(double)maxIndex());
    else return std::make_pair(false,0);
  }
  uint i=1;
  for (;i<slots_.size();++i)
    if (v<slots_[i]) break;

  if (useUnderFlow_) return std::make_pair(true,(double) i);
  //need to substract 1 because checking on upper edges
  else return std::make_pair(true,(double)i-1);
}


VariableComputer::VariableComputer(CachingVariable::CachingVariableFactoryArg arg){
  if (arg.iConfig.exists("separator"))
    separator_ = arg.iConfig.getParameter<std::string>("separator");
  else
    separator_ ="_";

  name_=arg.n;
  //get the configuration out
  std::string m = arg.iConfig.getParameter<std::string>("method");
  
  //prepare a list of purely virtual caching variable
  std::string aName;
  aName= arg.n+separator_+"toto";
  iCompute_["toto"] = new ComputedVariable(m,aName,arg.iConfig,this);
  arg.m[aName] = iCompute_["toto"];
    
  aName= arg.n+separator_+"tutu";
  iCompute_["tutu"] = new ComputedVariable(m,aName,arg.iConfig,this);
  arg.m[aName] = iCompute_["tutu"];

  aName= arg.n+separator_+"much";
  iCompute_["much"] = new ComputedVariable(m,aName,arg.iConfig,this);
  arg.m[aName] = iCompute_["much"];
}

void VariableComputer::compute(const edm::Event & iEvent) const{
  //does some mumbo jumbo with the event.
  // computes a bunch of doubles
  double toto = 3;
  double tutu = 4;
  
  //set the  variables  value (which do as if they had been cached)
  iCompute_.find("toto")->second->setCache(toto);
  iCompute_.find("tutu")->second->setCache(tutu);
  iCompute_.find("much")->second->setNotCompute();
}


ComputedVariable::ComputedVariable(CachingVariableFactoryArg arg ) : 
  CachingVariable("ComputedVariable",arg.n,arg.iConfig){
  // instanciate the computer
    std::string computerType = arg.iConfig.getParameter<std::string>("computer");
    myComputer = VariableComputerFactory::get()->create(computerType,arg);
    //    myComputer = new VariableComputer(arg);
    //there is a memory leak here, because the object we are in is not register anywhere. since it happens once per job, this is not a big deal.
}


CachingVariable::evalType Power::eval( const edm::Event & iEvent) const {
  const CachingVariable * var=edm::Service<VariableHelperService>()->get().variable(var_);
  if (!var->compute(iEvent)) return std::make_pair(false,0);

  double v=(*var)(iEvent);
  double p=exp(power_*log(v));
  return std::make_pair(true,p);
}

