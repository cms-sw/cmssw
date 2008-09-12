#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"
#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"



CachingVariable::evalType VarSplitter::eval(const edm::Event & iEvent) const{
  const CachingVariable * var=VariableHelperInstance::get().variable(var_);
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






CachingVariable::evalType Power::eval( const edm::Event & iEvent) const {
  const CachingVariable * var=VariableHelperInstance::get().variable(var_);
  if (!var->compute(iEvent)) return std::make_pair(false,0);

  double v=(*var)(iEvent);
  double p=exp(power_*log(v));
  return std::make_pair(true,p);
}
