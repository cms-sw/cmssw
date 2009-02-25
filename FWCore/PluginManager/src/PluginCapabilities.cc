// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginCapabilities
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Apr  6 12:36:24 EDT 2007
// $Id: PluginCapabilities.cc,v 1.4 2007/07/03 19:19:50 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edmplugin {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PluginCapabilities::PluginCapabilities()
{
  finishedConstruction();
}

// PluginCapabilities::PluginCapabilities(const PluginCapabilities& rhs)
// {
//    // do actual copying here;
// }

PluginCapabilities::~PluginCapabilities()
{
}

//
// assignment operators
//
// const PluginCapabilities& PluginCapabilities::operator=(const PluginCapabilities& rhs)
// {
//   //An exception safe implementation is
//   PluginCapabilities temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
bool 
PluginCapabilities::tryToFind(const SharedLibrary& iLoadable)
{
  void* sym;
  if( not iLoadable.symbol("SEAL_CAPABILITIES",sym) ) {
    return false;
  }
  
  const char** names;
  int size;
  //reinterpret_cast<void (*)(const char**&,int&)>(sym)(names,size);
  reinterpret_cast<void (*)(const char**&,int&)>(reinterpret_cast<unsigned long>(sym))(names,size);

  PluginInfo info;
  for(int i=0; i < size; ++i) {
    std::string name(names[i]);
    classToLoadable_[name]=iLoadable.path();
    
    //NOTE: can't use newPlugin(name) to do the work since it assumes
    //  we haven't yet returned from PluginManager::load method
    info.name_ = name;
    info.loadable_ = iLoadable.path();
    this->newPluginAdded_(category(),info);
  }
  return true;
}

void 
PluginCapabilities::load(const std::string& iName)
{
  if(classToLoadable_.end() == classToLoadable_.find(iName) ) {
    const SharedLibrary& lib = PluginManager::get()->load(category(),
                                                          iName);
    //read the items from the 'capabilities' symbol
    if(not tryToFind(lib) ) {
      throw cms::Exception("PluginNotFound")<<"The dictionary for class '"<<iName <<"' is supposed to be in file\n '"
      <<lib.path().native_file_string()<<"'\n but no dictionaries are in that file.\n"
      "It appears like the cache is wrong.  Please do 'EdmPluginRefresh "<<lib.path().native_file_string()<<"'.";
    }
    
    if(classToLoadable_.end() == classToLoadable_.find(iName)) {
      throw cms::Exception("PluginNotFound")<<"The dictionary for class '"<<iName<<"' is supposed to be in file\n '"
      <<lib.path().native_file_string()<<"'\n but was not found.\n"
      "It appears like the cache is wrong.  Please do 'EdmPluginRefresh "<<lib.path().native_file_string()<<"'.";
    }
  }
}

bool
PluginCapabilities::tryToLoad(const std::string& iName)
{
  if(classToLoadable_.end() == classToLoadable_.find(iName) ) {
    const SharedLibrary* lib = PluginManager::get()->tryToLoad(category(),
                                                          iName);
    if( 0 == lib) {
      return false;
    }
    //read the items from the 'capabilities' symbol
    if(not tryToFind(*lib) ) {
      throw cms::Exception("PluginNotFound")<<"The dictionary for class '"<<iName <<"' is supposed to be in file\n '"
      <<lib->path().native_file_string()<<"'\n but no dictionaries are in that file.\n"
      "It appears like the cache is wrong.  Please do 'EdmPluginRefresh "<<lib->path().native_file_string()<<"'.";
    }
    
    if(classToLoadable_.end() == classToLoadable_.find(iName)) {
      throw cms::Exception("PluginNotFound")<<"The dictionary for class '"<<iName<<"' is supposed to be in file\n '"
      <<lib->path().native_file_string()<<"'\n but was not found.\n"
      "It appears like the cache is wrong.  Please do 'EdmPluginRefresh "<<lib->path().native_file_string()<<"'.";
    }
  }
  return true;
}
//
// const member functions
//
std::vector<PluginInfo> 
PluginCapabilities::available() const
{
  PluginInfo info;
  std::vector<PluginInfo> infos;
  infos.reserve(classToLoadable_.size());
  
  for(std::map<std::string, boost::filesystem::path>::const_iterator it = classToLoadable_.begin();
      it != classToLoadable_.end();
      ++it) {
    info.name_ = it->first;
    info.loadable_ = it->second;
    infos.push_back(info);
  }
  return infos;
}

const std::string& 
PluginCapabilities::category() const
{
  static const std::string s_cat("Capability");
  return s_cat;
}

//
// static member functions
//
PluginCapabilities*
PluginCapabilities::get() {
  static PluginCapabilities s_instance;
  return &s_instance;
}

}
