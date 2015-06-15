#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"

class ExpressionEvaluatorCutWithEventContent : public CutApplicatorWithEventContentBase {
public:
  ExpressionEvaluatorCutWithEventContent(const edm::ParameterSet& c);
  
  result_type asCandidate(const argument_type& cand) const override final {
    return (*cut_)(cand);
  }

  void setConsumes(edm::ConsumesCollector& sumes) override final { 
    cut_->setConsumes(sumes);
  }

  void getEventContent(const edm::EventBase& event) override final { 
    cut_->getEventContent(event);
  }

private:
  std::unique_ptr<CutApplicatorWithEventContentBase> cut_;
};

ExpressionEvaluatorCutWithEventContent::
ExpressionEvaluatorCutWithEventContent(const edm::ParameterSet& c) : CutApplicatorWithEventContentBase(c) { 
  const std::string close_function(" };");
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
  cut_.reset(eval.expr<CutApplicatorWithEventContentBase>());

}


DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  ExpressionEvaluatorCutWithEventContent,
                  "ExpressionEvaluatorCutWithEventContent");
