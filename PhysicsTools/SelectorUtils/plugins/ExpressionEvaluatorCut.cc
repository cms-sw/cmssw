#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

class ExpressionEvaluatorCut : public CutApplicatorBase {
public:
  ExpressionEvaluatorCut(const edm::ParameterSet& c);
  virtual ~ExpressionEvaluatorCut(){};
  
  result_type asCandidate(const argument_type& cand) const override final {
    return (*cut_)(cand);
  }

  double value(const reco::CandidatePtr& cand) const override final {
    return cut_->value(cand);
  }

  const std::string& name() const override final { return realname_; }

private:
  const std::string realname_;
  CutApplicatorBase* cut_;
};

ExpressionEvaluatorCut::
ExpressionEvaluatorCut(const edm::ParameterSet& c) : 
  CutApplicatorBase(c),
  realname_(c.getParameter<std::string>("realCutName"))
{
  const std::string newline("\n");
  const std::string close_function("; };");
  const std::string candTypePreamble("CandidateType candidateType() const override final { return ");
  
  //construct the overload of candidateType()
  const std::string& candType = c.getParameter<std::string>("candidateType");
  const std::string candTypeExpr = candTypePreamble + candType + close_function;
  
  // read in the overload of operator()
  const std::string& oprExpr = c.getParameter<std::string>("functionDef");
  
  // read in the overload of value()
  const std::string& valExpr = c.getParameter<std::string>("valueDef");

  // concatenate and evaluate the expression
  const std::string total_expr = ( candTypeExpr + newline + 
                                   oprExpr      + newline +
                                   valExpr                  );
  reco::ExpressionEvaluator eval("PhysicsTools/SelectorUtils",
                                 "CutApplicatorBase",
                                 total_expr.c_str());
  cut_ = eval.expr<CutApplicatorBase>();
}

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  ExpressionEvaluatorCut,
                  "ExpressionEvaluatorCut");
