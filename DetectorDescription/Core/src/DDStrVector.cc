
#include "DetectorDescription/Core/interface/DDStrVector.h"
#include "DetectorDescription/Base/interface/DDException.h"

// Evaluator 
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDStrVector::DDStrVector() : DDBase<DDName,std::vector<std::string>*>() { }


DDStrVector::DDStrVector(const DDName & name) : DDBase<DDName,std::vector<std::string>*>() 
{
  prep_ = StoreT::instance().create(name);
}

DDStrVector::DDStrVector(const DDName & name,std::vector<std::string>* vals)
{
  prep_ = StoreT::instance().create(name,vals);
}  


std::ostream & operator<<(std::ostream & os, const DDStrVector & cons)
{
  os << "DDStrVector name=" << cons.name(); 
  
  if(cons.isDefined().second) {
    os << " size=" << cons.size() << " vals=( ";
    DDStrVector::value_type::const_iterator it(cons.values().begin()), ed(cons.values().end());
    for(; it<ed; ++it) {
      os << *it << ' ';
    }
    os << ')';
  }
  else {
    os << " constant is not yet defined, only declared.";
  }  
  return os;
}

