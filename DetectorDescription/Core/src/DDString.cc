#include "DetectorDescription/Core/interface/DDString.h"

#include <utility>

DDString::DDString() : DDBase< DDName, std::string*>() { }

DDString::DDString( const DDName & name )
  : DDBase< DDName, std::string* >() 
{
  create( name );
}

DDString::DDString( const DDName & name, std::string* vals )
{
  create( name, vals );
}  

std::ostream & operator<<( std::ostream & os, const DDString & cons )
{
  os << "DDString name=" << cons.name(); 
  
  if( cons.isDefined().second ) {
    os << " val=" << cons.value();
  }
  else {
    os << " constant is not yet defined, only declared.";
  }  
  return os;
}
