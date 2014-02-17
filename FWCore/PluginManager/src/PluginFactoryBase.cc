// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginFactoryBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 13:09:50 EDT 2007
// $Id: PluginFactoryBase.cc,v 1.8 2011/08/24 12:28:53 eulisse Exp $
//

// system include files

// user include files
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/PluginFactoryManager.h"


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

// PluginFactoryBase::PluginFactoryBase(const PluginFactoryBase& rhs)
// {
//    // do actual copying here;
// }

PluginFactoryBase::~PluginFactoryBase()
{
}

//
// assignment operators
//
// const PluginFactoryBase& PluginFactoryBase::operator=(const PluginFactoryBase& rhs)
// {
//   //An exception safe implementation is
//   PluginFactoryBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
PluginFactoryBase::finishedConstruction()
{
   PluginFactoryManager::get()->addFactory(this);
}

void
PluginFactoryBase::newPlugin(const std::string& iName)
{
  PluginInfo info;
  info.loadable_=boost::filesystem::path(PluginManager::loadingFile());
  info.name_=iName;
  newPluginAdded_(category(),info);
}


PluginFactoryBase::Plugins::const_iterator 
PluginFactoryBase::findPMaker(const std::string& iName) const
{
  //do we already have it?
  Plugins::const_iterator itFound = m_plugins.find(iName);
  if(itFound == m_plugins.end()) {
    std::string lib = PluginManager::get()->load(this->category(),iName).path().string();
    itFound = m_plugins.find(iName);
    if(itFound == m_plugins.end()) {
      throw cms::Exception("PluginCacheError")<<"The plugin '"<<iName<<"' should have been in loadable\n '"
      <<lib<<"'\n but was not there.  This means the plugin cache is incorrect.  Please run 'EdmPluginRefresh "<<lib<<"'";
    }
  } else {
    checkProperLoadable(iName,itFound->second.front().second);
  }
  return itFound;
}


PluginFactoryBase::Plugins::const_iterator 
PluginFactoryBase::tryToFindPMaker(const std::string& iName) const
{
  //do we already have it?
  Plugins::const_iterator itFound = m_plugins.find(iName);
  if(itFound == m_plugins.end()) {
    const SharedLibrary* slib = PluginManager::get()->tryToLoad(this->category(),iName);
    if(0!=slib) {
      std::string lib = slib->path().string();
      itFound = m_plugins.find(iName);
      if(itFound == m_plugins.end()) {
        throw cms::Exception("PluginCacheError")<<"The plugin '"<<iName<<"' should have been in loadable\n '"
        <<lib<<"'\n but was not there.  This means the plugin cache is incorrect.  Please run 'EdmPluginRefresh "<<lib<<"'";
      }
    }
  } else {
    checkProperLoadable(iName,itFound->second.front().second);
  }
  return itFound;
}

void 
PluginFactoryBase::fillInfo(const PMakers &makers,
              PluginInfo& iInfo,
              std::vector<PluginInfo>& iReturn ) const {
  for(PMakers::const_iterator it = makers.begin();
      it != makers.end();
      ++it) {
    iInfo.loadable_ = it->second;
    iReturn.push_back(iInfo);
  }
}

void 
PluginFactoryBase::fillAvailable(std::vector<PluginInfo>& iReturn) const {
  PluginInfo info;
  for( Plugins::const_iterator it = m_plugins.begin();
      it != m_plugins.end();
      ++it) {
    info.name_ = it->first;
    fillInfo(it->second,
             info, iReturn);
  }
}

void 
PluginFactoryBase::checkProperLoadable(const std::string& iName, const std::string& iLoadedFrom) const
{
  //should check to see if this is from the proper loadable if it
  // was not statically linked
  if (iLoadedFrom != PluginManager::staticallyLinkedLoadingFileName() &&
      PluginManager::isAvailable()) {
    if( iLoadedFrom != PluginManager::get()->loadableFor(category(),iName).string() ) {
      throw cms::Exception("WrongPluginLoaded")<<"The plugin '"<<iName<<"' should have been loaded from\n '"
      <<PluginManager::get()->loadableFor(category(),iName).string()
      <<"'\n but instead it was already loaded from\n '"
      <<iLoadedFrom<<"'\n because some other plugin was loaded from the latter loadables.\n"
      "To work around the problem the plugin '"<<iName<<"' should only be defined in one of these loadables.";
    }
  }
}


void 
PluginFactoryBase::registerPMaker(void* iPMaker, const std::string& iName) {
  m_plugins[iName].push_back(std::pair<void*,std::string>(iPMaker,PluginManager::loadingFile()));
  newPlugin(iName);
}

std::vector<PluginInfo> 
PluginFactoryBase::available() const {
  std::vector<PluginInfo> returnValue;
  returnValue.reserve(m_plugins.size());
  fillAvailable(returnValue);
  return returnValue;
}

//
// const member functions
//

//
// static member functions
//
}
