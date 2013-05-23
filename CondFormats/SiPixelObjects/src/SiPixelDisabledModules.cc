// SiPixelDisabledModules.cc
//
// class implementation to hold a list of disabled pixel modules
//
// M. Eads
// Apr 2008

#include "CondFormats/SiPixelObjects/interface/SiPixelDisabledModules.h"

#include <algorithm>

// add a list of modules to the vector of disabled modules
void SiPixelDisabledModules::addDisabledModule(const disabledModuleListType& idVector) {
  theDisabledModules.insert(theDisabledModules.end(),
			    idVector.begin(),
			    idVector.end());

} // void SiPixelDisabledModules::addDisabledModule(disabledModuleListType idVector)


// remove disabled module from the list
// returns false if id not in disable list, true otherwise
bool SiPixelDisabledModules::removeDisabledModule(disabledModuleType module) {
  disabledModuleListType::iterator iter = find(theDisabledModules.begin(),
					      theDisabledModules.end(),
					      module);
  if (iter == theDisabledModules.end())
    return false;
  
  theDisabledModules.erase(iter);
  return true;

} // bool SiPixelDisabledModules::removeDisabledModule(disabledModuleType module)


// method to return true if the specified module is in the list
// of disabled modules
bool SiPixelDisabledModules::isModuleDisabled(disabledModuleType module) {
  disabledModuleListType::const_iterator iter = find(theDisabledModules.begin(),
						     theDisabledModules.end(),
						     module);

  return iter != theDisabledModules.end();

} // bool SiPixelDisabledModules::isModuleDisabled(disabledModuleType module)

