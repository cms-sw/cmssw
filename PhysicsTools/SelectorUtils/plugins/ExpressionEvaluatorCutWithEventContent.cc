#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"

class ExpressionEvaluatorCutWithEventContent : public CutApplicatorWithEventContentBase {
public:
  ExpressionEvaluatorCutWithEventContent(const edm::ParameterSet& c);
  virtual ~ExpressionEvaluatorCutWithEventContent() {};

  result_type asCandidate(const argument_type& cand) const override final {
    return (*cut_)(cand);
  }

  void setConsumes(edm::ConsumesCollector& sumes) override final { 
    cut_->setConsumes(sumes);
  }

  void getEventContent(const edm::EventBase& event) override final { 
    cut_->getEventContent(event);
  }

  double value(const reco::CandidatePtr& cand) const override final {
    return cut_->value(cand);
  }

  const std::string& name() const override final { return realname_; }

private:
  const std::string realname_;
  CutApplicatorWithEventContentBase* cut_;
};

ExpressionEvaluatorCutWithEventContent::
ExpressionEvaluatorCutWithEventContent(const edm::ParameterSet& c) : 
  CutApplicatorWithEventContentBase(c),
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

  // read in the overload of setConsumes()
  const std::string& setConsumesExpr = 
    c.getParameter<std::string>("setConsumesDef");

  // read in the overload of getEventContent()
  const std::string& getEventContentExpr = 
    c.getParameter<std::string>("getEventContentDef");
  

  // concatenate and evaluate the expression
  const std::string total_expr = ( candTypeExpr        + newline + 
                                   oprExpr             + newline +
                                   valExpr             + newline +
                                   setConsumesExpr     + newline +
                                   getEventContentExpr             );
  reco::ExpressionEvaluator eval("PhysicsTools/SelectorUtils",
                                 "CutApplicatorWithEventContentBase",
                                 total_expr.c_str());
  cut_ = eval.expr<CutApplicatorWithEventContentBase>();

}


DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  ExpressionEvaluatorCutWithEventContent,
                  "ExpressionEvaluatorCutWithEventContent");
