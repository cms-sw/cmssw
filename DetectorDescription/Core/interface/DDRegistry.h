#ifndef DDD_DDRegistry_h
#define DDD_DDRegistry_h

#include <map>
//#include "DetectorDescription/Base/interface/Singleton.h"
#include "DetectorDescription/Core/interface/DDName.h"

template <class T>
class DDRegistry : public std::map<DDName,T>
{
public:
/*
  typedef std::map<DDName,T> RegistryMap;
  typedef typename RegistryMap::iterator iterator;
  typedef typename RegistryMap::const_iterator const_iterator;
  typedef typename RegistryMap::value_type value_type;
  typedef typename RegistryMap::key_type key_type;
  typedef typename RegistryMap::std::mapped_type std::mapped_type;
*/  
private:
  //RegMap reg_;  	
};
#endif //. DDD_DDRegistry_h
