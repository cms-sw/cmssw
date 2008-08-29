#ifndef _VariableEventSelector_H
#define _VariableEventSelector_H

#include "PhysicsTools/UtilAlgos/interface/EventSelector.h"
#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"

class VariableEventSelector : public EventSelector {
 public:
  VariableEventSelector(const edm::ParameterSet& pset) :
    EventSelector(pset)
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
