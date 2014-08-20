#ifndef PhysicsToolsPatUtils_RazorComputer_H
#define PhysicsToolsPatUtils_RazorComputer_H
#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"


class RazorBox : public CachingVariable {
 public:
  RazorBox(const CachingVariable::CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC) ;
  ~RazorBox(){}
  
  void compute(const edm::Event & iEvent) const;
 private:
  double par_;
};

class RazorComputer : public VariableComputer {
 public:
  RazorComputer(const CachingVariable::CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC) ;
  ~RazorComputer(){};

  void compute(const edm::Event & iEvent) const;
 private:
  edm::InputTag jet_;
  edm::InputTag met_;
  float pt_,eta_;
  
};

#endif
