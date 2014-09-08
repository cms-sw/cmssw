#ifndef _VariableEventSelector_H
#define _VariableEventSelector_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/EventSelector.h"
#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"
#include "TFormula.h"
#include "TString.h"

class VariableFormulaEventSelector : public EventSelector {
 public: VariableFormulaEventSelector(const edm::ParameterSet& pset, edm::ConsumesCollector && iC) :
  VariableFormulaEventSelector(pset, iC) {}
 VariableFormulaEventSelector(const edm::ParameterSet& pset, edm::ConsumesCollector & iC) :
  EventSelector(pset, iC)
    {
      const std::string name_ = pset.getParameter<std::string>("fname");
      TString ts(pset.getParameter<std::string>("formula"));
      //find the variables. register and replace
      int open = ts.Index("[");
      int close = ts.Index("]");
      while (open>0){
	++open;
	TString sub( ts(open,close-open));
	//std::cout<<"found:"<< sub <<std::endl;
	vars_.insert( sub.Data() );
	//vars_.insert(  ts(open,close-open));
	open = ts.Index("[",open);
	close = ts.Index("]",open);
      }
      
      unsigned int v_i;
      std::set<std::string>::iterator v_it;
      for (v_i = 0, v_it=vars_.begin();
	   v_i!=vars_.size(); ++v_i,++v_it)
	{
	  ts.ReplaceAll(TString::Format("[%s]", v_it->c_str()),TString::Format("[%d]", v_i));
	}

      //std::cout<<" formula found:"<< ts <<std::endl;
      formula_ = new TFormula(name_.c_str(), ts);
      //vars_ = pset.getParameter<std::vector<std::string>>("variables");
      threshold_= pset.getParameter<double>("threshold");
    }

  bool select(const edm::Event& e) const{
    unsigned int v_i;
    std::set<std::string>::iterator v_it;

    for (v_i = 0, v_it=vars_.begin();
	 v_i!=vars_.size(); ++v_i,++v_it)
      {
	const CachingVariable * var = edm::Service<VariableHelperService>()->get().variable(*v_it);
	if (!var->compute(e)) return false;
	double v=(*var)(e);
	formula_->SetParameter(v_i, v);	
      }

    //should be valuated 0. or 1. in double
    return (formula_->Eval(0.)>=threshold_);
  }
 private:
  //std::string formula_;
  TFormula * formula_;
  std::set<std::string> vars_;
  double threshold_;
};

class VariableEventSelector : public EventSelector {
 public:
  VariableEventSelector(const edm::ParameterSet& pset, edm::ConsumesCollector && iC) :
    VariableEventSelector(pset, iC) {}
  VariableEventSelector(const edm::ParameterSet& pset, edm::ConsumesCollector & iC) :
    EventSelector(pset, iC)
    {
      var_=pset.getParameter<std::string>("var");
      doMin_=pset.exists("min");
      if (doMin_) min_=pset.getParameter<double>("min");
      doMax_=pset.exists("max");
      if (doMax_) max_=pset.getParameter<double>("max");

      std::stringstream ss;
      ss<<"event selector based on VariableHelper variable: "<<var_;
      description_.push_back(ss.str());       ss.str("");
      if (doMin_){
	ss<<"with minimum boundary: "<<min_;
	description_.push_back(ss.str());       ss.str("");}
      if (doMax_){
	ss<<"with maximum boundary: "<<max_;
	description_.push_back(ss.str());       ss.str("");}
    }
    bool select(const edm::Event& e) const{
      const CachingVariable * var=edm::Service<VariableHelperService>()->get().variable(var_);
      if (!var->compute(e)) return false;

      double v=(*var)(e);

      if (doMin_ && v<min_) return false;
      else if (doMax_ && v>max_) return false;
      else return true;
    }

 private:
  std::string var_;
  bool doMin_;
  double min_;
  bool doMax_;
  double max_;
};

#endif
