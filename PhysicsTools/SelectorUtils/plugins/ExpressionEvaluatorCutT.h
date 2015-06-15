#ifndef __PhysicsTools_SelectorUtils_ExpressionEvaluatorCutT_H__
#define __PhysicsTools_SelectorUtils_ExpressionEvaluatorCutT_H__


#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

template<class Base>
class ExpressionEvaluatorCutT : public Base {
public:
  ExpressionEvaluatorCutT(const edm::ParameterSet& c);
  
  result_type asCandidate(const argument_type& cand) const override final {
    return (*cut_)(cand);
  }

private:
  std::unique_ptr<Base> cut_;
};

template<class Base>
ExpressionEvaluatorCutT::
ExpressionEvaluatorCutT(const edm::ParameterSet& c) :
    Base(c) { 
  
}

#endif
