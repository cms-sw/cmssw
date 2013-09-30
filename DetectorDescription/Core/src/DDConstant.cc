#include "DetectorDescription/Core/interface/DDConstant.h"

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


void DDConstant::createConstantsFromEvaluator(ClhepEvaluator &evaluator)
{
  const std::vector<std::string> & vars = evaluator.variables();
  const std::vector<std::string> & vals = evaluator.values();
  if (vars.size() != vals.size()) {
    throw cms::Exception("DDException") << "DDConstants::createConstansFromEvaluator(): different size of variable names & values!";
  }
  size_t i(0), s(vars.size());
  for (; i<s; ++i) {
    const std::string & sr = vars[i];
    typedef std::string::size_type ST;
    ST i1 = sr.find("___");
    DDName name(std::string(sr,i1+3,sr.size()-1),std::string(sr,0,i1));       
    double* dv = new double;
    *dv = evaluator.eval(sr.c_str());
    DDConstant cst(name,dv);//(ddname); 
  }
}
