#include "DetectorDescription/Core/interface/DDVector.h"

#include <utility>

DDVector::DDVector()
  : DDBase< DDName, std::vector< double >*>()
{}

DDVector::DDVector( const DDName & name )
  : DDBase< DDName, std::vector< double >*>() 
{
  create( name );
}

DDVector::DDVector( const DDName & name, std::vector<double>* vals )
{
  create( name, vals );
}  

std::ostream & operator<<(std::ostream & os, const DDVector & cons)
{
  os << "DDVector name=" << cons.name(); 
  
  if(cons.isDefined().second) {
    os << " size=" << cons.size() << " vals=( ";
    DDVector::value_type::const_iterator it(cons.values().begin()), ed(cons.values().end());
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




DDVector::operator std::vector<int>() const
{
   std::vector<int> result(rep().size());
   std::vector<int>::size_type sz=0;
   std::vector<double>::const_iterator it(rep().begin()), ed(rep().end());
   for (; it != ed; ++it) { 
     result[sz] = int(*it);
     ++sz;
   }  
   return result;
}
