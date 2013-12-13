// -*- C++ -*-
//
// Package:     LibraryLoader
// Class  :     RootAutoLibraryLoader
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Wed Nov 30 14:55:01 EST 2005
//

#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"

#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/RootAutoLibraryLoader/src/stdNamespaceAdder.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"

#include "TClass.h"
#include "TInterpreter.h"
#include "TROOT.h"

#include <map>
#include <string>

using CallbackPtr = int (*)(const char*);
static void* gPrevious = nullptr;

/// Check if the class name is known to the ROOT interpreter.
static
bool
interpreterLookupClass(const std::string& name)
{
  ClassInfo_t* ci = gInterpreter->ClassInfo_Factory(name.c_str());
  return gInterpreter->ClassInfo_IsValid(ci);
}

/// Use the SEAL Capabilities plugin to load a library which
/// provides the definition of a class, and verify that the
/// ROOT interpreter now knows about the class.
static
bool
loadLibraryForClass(const char* classname)
{
  if (classname == nullptr) {
    return false;
  }
  const std::string& cPrefix = edm::dictionaryPlugInPrefix();
  std::string name(classname);
  try {
    if (edmplugin::PluginCapabilities::get()->tryToLoad(cPrefix + name)) {
      // We loaded a library, now check to see if we got the class.
      if (!interpreterLookupClass(name)) {
        // Nope, did not get it.
        return false;
      }
      // Good, library loaded for class and the interpreter knows it now.
      return true;
    }
    // Try adding a std namespace qualifier.
    std::string stdName = edm::root::stdNamespaceAdder(name);
    if (!edmplugin::PluginCapabilities::get()->tryToLoad(cPrefix + stdName)) {
      // Bad, we did not load a library for the class.
      return false;
    }
    // We loaded a library, now check to see if we got the class.
    if (!interpreterLookupClass(stdName)) {
      // Nope, did not get it.
      return false;
    }
    // Good, library loaded for class and the interpreter knows it now.
    return true;
  }
  catch (cms::Exception& e) {
    // Bad, loading threw a known CMS exception.
    return false;
  }
  // Note: Cannot happen!
  return true;
}

/// Callback for the ROOT interpreter to invoke when it fails
/// during lookup of a class name.  We try to load a library
/// that provides the definition of the class using the SEAL
/// Capabilities plugin.
#include <iostream>
static
int
AutoLoadCallback(const char* c)
{
  if (c == nullptr) {
    return 0;
  }
  int result = loadLibraryForClass(c);
  if (result) {
    // Good, library loaded and class is now known by the interpreter.
    return result;
  }
  // Failed, chain to next callback.
  if (gPrevious) {
    result = (*reinterpret_cast<CallbackPtr>(gPrevious))(c);
  }
  return result;
}

#if 0

static
void
registerTypes()
{
  //
  // Setup the CINT autoloading table so that it knows which
  // library to load for every SEAL Capability dictionary
  // plugin class.
  //
  //--
  // Get the list of SEAL Capability plugins from the plugin manager.
  edmplugin::PluginManager *db = edmplugin::PluginManager::get();
  auto itFound = db->categoryToInfos().find("Capability");
  if (itFound == db->categoryToInfos().end()) {
    return;
  }
  std::string const& cPrefix = edm::dictionaryPlugInPrefix();
  using ClassAndLibraries = std::vector<std::pair<std::string, std::string>>;
  ClassAndLibraries classesAndLibs;
  classesAndLibs.reserve(1000);
  {
    std::string prev;
    // Loop over all the plugins.
    for (auto const & info : itFound->second) {
      if (prev == info.name_) {
        continue;
      }
      prev = info.name_;
      if (info.name_.substr(0, cPrefix.size() != cPrefix)) {
        // Ignore plugins that are not dictionary plugins.
        continue;
      }
      std::string className(info.name_, cPrefix.size());
      classesAndLibs.emplace_back(className, info.loadable_.string());
    }
  }
  // FIXME: Probably not needed anymore.
  // In order to determine if a name is from a class or a
  // namespace, we will order all the classes in descending
  // order so that embedded classes will be seen before their
  // containing classes, that way we can say the containing
  // class is a namespace before finding out it is actually
  // a class.
  for (auto I = classesAndLibs.rbegin(), E = classesAndLibs.rend(); I != E;
      ++I) {
    const std::string& className = I->first;
    const std::string& libraryName = I->second;
    // We need to register namespaces and to figure out if we
    // have an embedded class.
    static const std::string toFind(":<");
    auto pos = className.find_first_of(toFind, pos);
    while (pos != std::string::npos) {
      // We are at a namespace boundary, or a template argument boundary.
      if (className[pos] == '<') {
        // Template argument boundary, no need to scan further
        // for namespace names, and we now register the whole
        // class name (Note: We may do this multiple times!).
        break;
      }
      if ((className.size() <= (pos + 1)) || (className[pos+1] != ':')) {
        break;
      }
      // FIXME: This does bad things if a template argument is qualified!
      // FIXME: if className.substr(0, pos).find('<') { bad }
      // We have found a "::" in the name, register a namespace.
      G__set_class_autoloading_table(className.substr(0, pos).c_str(), "");
      pos += 2;
    }
    G__set_class_autoloading_table(className.c_str(), libraryName.c_str());
    // Continue scanning from here.
    pos = className.find_first_of(toFind, pos);
  }
}
#endif // 0

namespace edm {

RootAutoLibraryLoader::
RootAutoLibraryLoader()
  : classNameAttemptingToLoad_()
{
  // Note: The only thing AssertHandler really does is call
  // PluginManager::configure() with the default configuration.
  AssertHandler h;
  // Register our ROOT TClass::GetClass(std::type_info) and
  // TROOT::LoadClass(name, silent) hook.
  gROOT->AddClassGenerator(this);
  // Register our ROOT interpreter class-name-not-found hook.
  gPrevious = gInterpreter->GetAutoLoadCallBack();
  gInterpreter->SetAutoLoadCallBack(reinterpret_cast<void*>(&AutoLoadCallback));
  //registerTypes();
}

TClass*
RootAutoLibraryLoader::
GetClass(char const* classname, bool load)
{
  if (!strcmp(classname, classNameAttemptingToLoad_.c_str())) {
    // Recursive call to ourselves detected. We could not load
    // the class and ROOT could not load the class.
    return nullptr;
  }
  if (!load) {
    return nullptr;
  }
  // Try to load a library which provides the class definition
  // using the SEAL Capability plugin.
  if (!loadLibraryForClass(classname)) {
    // Bad, we could not load a library.
    return nullptr;
  }
  // Good, library was loaded, now have ROOT provide the TClass.
  // Note: Here we guard against a recursive call to ourselves.
  classNameAttemptingToLoad_ = classname;
  auto ret = gROOT->GetClass(classname, true);
  classNameAttemptingToLoad_.clear();
  return ret;
}

TClass*
RootAutoLibraryLoader::
GetClass(const type_info& typeinfo, bool load)
{
  if (!load) {
    return nullptr;
  }
  // FIXME: demangle the name here!
  std::string demangledName = typeDemangle(typeinfo.name());
  return GetClass(demangledName.c_str(), load);
}

void
RootAutoLibraryLoader::
enable()
{
  static RootAutoLibraryLoader singleton;
}

void
RootAutoLibraryLoader::
loadAll()
{
  enable();
  auto db = edmplugin::PluginManager::get();
  auto itFound = db->categoryToInfos().find("Capability");
  if (itFound == db->categoryToInfos().end()) {
    return;
  }
  std::string prev;
  for (auto const& plugin_info : itFound->second) {
    if (plugin_info.name_ == prev) {
      continue;
    }
    edmplugin::PluginCapabilities::get()->load(plugin_info.name_);
    prev = plugin_info.name_;
  }
}

} // namespace edm

