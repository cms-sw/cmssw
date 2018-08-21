#include "DetectorDescription/Core/interface/DDStrVector.h"

#include <utility>

DDStrVector::DDStrVector()
  : DDBase< DDName, std::vector<std::string>*>()
{}

DDStrVector::DDStrVector( const DDName & name )
  : DDBase< DDName, std::vector< std::string >*>() 
{
  create( name );
}

DDStrVector::DDStrVector( const DDName & name, std::vector<std::string>* vals )
{
  create( name, vals );
}

std::ostream & operator<<( std::ostream & os, const DDStrVector & cons )
{
  os << "DDStrVector name=" << cons.name(); 
  
  if( cons.isDefined().second ) {
    os << " size=" << cons.size() << " vals=( ";
    for( const auto& it : cons.values()) {
      os << it << ' ';
    }
    os << ')';
  }
  else {
    os << " constant is not yet defined, only declared.";
  }  
  return os;
}
