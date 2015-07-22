#include "DetectorDescription/Core/interface/DDMap.h"

DDMap::DDMap() : DDBase<DDName,dd_map_type*>() { }

DDMap::DDMap(const DDName & name) : DDBase<DDName,dd_map_type*>() 
{
  prep_ = StoreT::instance().create(name);
}

DDMap::DDMap(const DDName & name,dd_map_type* vals)
{
  prep_ = StoreT::instance().create(name,vals);
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


