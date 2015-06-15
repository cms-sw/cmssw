#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

class ExpressionEvaluatorCut : public CutApplicatorBase {
public:
  ExpressionEvaluatorCut(const edm::ParameterSet& c);
  
  result_type asCandidate(const argument_type& cand) const override final {
    return (*cut_)(cand);
  }

private:
  std::unique_ptr<CutApplicatorBase> cut_;
};

ExpressionEvaluatorCut::
ExpressionEvaluatorCut(const edm::ParameterSet& c) : CutApplicatorBase(c) { 
  const std::string close_function("; };");
  const std::string candTypePreamble("CandidateType candidateType() const override final { return ");
  
  //construct the overload of candidateType()
  const std::string& candType = c.getParameter<std::string>("candidateType");
  const std::string candTypeExpr = candTypePreamble + candType + close_function;
  
  // read in the overload of operator()
  const std::string& oprExpr = c.getParameter<std::string>("functionDef");
  
  // concatenate and evaluate the expression
  const std::string total_expr = candTypeExpr + std::string("\n") + oprExpr;
  reco::ExpressionEvaluator eval("PhysicsTools/SelectorUtils",
                                 "CutApplicatorBase",
                                 total_expr.c_str());
  cut_.reset(eval.expr<CutApplicatorBase>());

}

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  ExpressionEvaluatorCut,
                  "ExpressionEvaluatorCut");
