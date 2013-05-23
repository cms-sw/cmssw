// SiPixelDisabledModules.h
//
// class definition to hold a list of disabled pixel modules
//
// M. Eads
// Apr 2008

#ifndef SiPixelDisabledModules_H
#define SiPixelDisabledModules_H

#include <vector>
#include <utility>

#include "DataFormats/DetId/interface/DetId.h"

class SiPixelDisabledModules {

 public:
  typedef DetId disabledModuleType;
  typedef std::vector<disabledModuleType> disabledModuleListType;

  SiPixelDisabledModules() {;}

  // constructor from a list of disabled modules
  SiPixelDisabledModules(const disabledModuleListType& disabledModules) : theDisabledModules(disabledModules) {;}

  virtual ~SiPixelDisabledModules() {;}

  // return the list of disabled modules/ROCs
  disabledModuleListType getDisabledModuleList()
    { return theDisabledModules; }

  // set the list of disabled modules (current list is lost)
  void setDisabledModuleList(const disabledModuleListType& disabledModules)
    { theDisabledModules = disabledModules; }

  // add a single module to the vector of disabled modules
  void addDisabledModule(disabledModuleType module)
  { theDisabledModules.push_back(module); }

  // add a vector of modules to the vector of disabled modules
  void addDisabledModule(const disabledModuleListType& idVector);

  // remove disabled module from the list
  // returns false if id not in disable list, true otherwise
  bool removeDisabledModule(disabledModuleType module);

  // check if a particular module is in the disabled list
  // return true if it is
  bool isModuleDisabled(disabledModuleType module);

 private:
  disabledModuleListType theDisabledModules;

}; // class SiPixelDisabledModules










#endif
