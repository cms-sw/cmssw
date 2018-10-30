#include "DetectorDescription/Core/interface/DDMap.h"

#include <utility>

DDMap::DDMap()
  : DDBase< DDName, std::unique_ptr<dd_map_type>>() { }

DDMap::DDMap( const DDName & name )
  : DDBase< DDName, std::unique_ptr<dd_map_type>>() 
{
  create( name );
}

DDMap::DDMap( const DDName & name, std::unique_ptr<dd_map_type> vals )
{
  create( name, std::move( vals ));
}

std::ostream & operator<<(std::ostream & os, const DDMap & cons)
{
  os << "DDMap name=" << cons.name(); 
  
  if(cons.isDefined().second) {
    os << " size=" << cons.size() << " vals=( ";
    for( const auto& it : cons.values()) {
      os << it.first << '=' << it.second << ' ';
    }
    os << ')';
  }
  else {
    os << " DDMap is not yet defined, only declared.";
  }  
  return os;
}
