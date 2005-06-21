#ifndef DDD_DDRegistry_h
#define DDD_DDRegistry_h

#include <map>
//#include "DetectorDescription/DDBase/interface/Singleton.h"
#include "DetectorDescription/DDCore/interface/DDName.h"

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
  typedef typename RegistryMap::mapped_type mapped_type;
*/  
private:
  //RegMap reg_;  	
};
#endif //. DDD_DDRegistry_h
