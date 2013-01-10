
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Base/interface/DDException.h"

// Evaluator 
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDConstant::DDConstant() : DDBase<DDName,double*>() { }


DDConstant::DDConstant(const DDName & name) : DDBase<DDName,double*>() 
{
  prep_ = StoreT::instance().create(name);
}

DDConstant::DDConstant(const DDName & name,double* vals)
{
  prep_ = StoreT::instance().create(name,vals);
}  


std::ostream & operator<<(std::ostream & os, const DDConstant & cons)
{
  os << "DDConstant name=" << cons.name(); 
  
  if(cons.isDefined().second) {
    os << " val=" << cons.value();
  }
  else {
    os << " constant is not yet defined, only declared.";
  }  
  return os;
}


void DDConstant::createConstantsFromEvaluator()
{
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  ClhepEvaluator * eval = dynamic_cast<ClhepEvaluator*>(&ev);
  if (eval){
    const std::vector<std::string> & vars = eval->variables();
    const std::vector<std::string> & vals = eval->values();
    if (vars.size() != vals.size()) {
      throw DDException("DDConstants::createConstansFromEvaluator(): different size of variable names & values!") ;
    }
    size_t i(0), s(vars.size());
    for (; i<s; ++i) {
      const std::string & sr = vars[i];
      typedef std::string::size_type ST;
      ST i1 = sr.find("___");
      DDName name(std::string(sr,i1+3,sr.size()-1),std::string(sr,0,i1));       
      double* dv = new double;
      *dv = eval->eval(sr.c_str());
      DDConstant cst(name,dv);//(ddname); 
    }  
  }
  else {
    throw DDException("DDConstants::createConstansFromEvaluator(): expression-evaluator is not a ClhepEvaluator-implementation!");
  }
}


