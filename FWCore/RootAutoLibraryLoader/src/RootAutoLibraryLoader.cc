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
// $Id: RootAutoLibraryLoader.cc,v 1.13 2006/10/21 02:48:58 wmtan Exp $
//

// system include files
#include <iostream>
#include "TROOT.h"
#include "G__ci.h"
#include "common.h"
#include "boost/regex.hpp"

// user include files
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/RootAutoLibraryLoader/src/stdNamespaceAdder.h"

#include "PluginManager/PluginManager.h"
#include "PluginManager/ModuleCache.h"
#include "PluginManager/Module.h"
#include "PluginManager/PluginCapabilities.h"

#include "Reflex/Type.h"
#include "Cintex/Cintex.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//
//hold onto the previous autolibrary loader
typedef int (*CallbackPtr) G__P((char*,char*));
static CallbackPtr gPrevious = 0;
static const char* kDummyLibName = "*dummy";

//This is actually defined within ROOT's v6_struct.cxx file but is not declared static
// I want to use it so that if the autoloading is already turned on, I can call the previously declared routine
extern CallbackPtr G__p_class_autoloading;

namespace edm {

static 
bool loadLibraryForClass(const char* classname)
{  
  //std::cout <<"loadLibaryForClass"<<std::endl;
  if(0 == classname) {
    return false;
  }
  //std::cout <<"asking to find "<<classname<<std::endl;
  static const std::string cPrefix("LCGReflex/");
  //std::cout <<"asking to find "<<cPrefix+classname<<std::endl;
  seal::PluginCapabilities::get()->load(cPrefix+classname);
  
  ROOT::Reflex::Type t = ROOT::Reflex::Type::ByName(classname);
  if(ROOT::Reflex::Type() != t) {
     if(!t.IsComplete()) {
	// this message happens too often (to many false positives) to be useful plus ROOT will complain about a missing dictionary 
	//std::cerr <<"Warning: Reflex knows about type '"<<classname<<"' but has no dictionary for it."<<std::endl;
	return false;
     }
    //std::cout <<"loaded "<<classname<<std::endl;
    return true;
  } 
  //see if adding a std namespace helps
  std::string name = root::stdNamespaceAdder(classname);
  //std::cout <<"see if std helps"<<std::endl;
  seal::PluginCapabilities::get()->load(cPrefix+name);
  
  t = ROOT::Reflex::Type::ByName(classname);
  return ROOT::Reflex::Type() != t;
}

//Based on code in ROOT's TCint.cxx file

static int ALL_AutoLoadCallback(char *c, char *l) {
  //NOTE: if the library (i.e. 'l') is an empty string this means we are dealing with a namespace
  // These checks appear to avoid a crash of ROOT during shutdown of the application
  if(0==c || 0==l || l[0]==0) {
    return 0;
  }
  ULong_t varp = G__getgvp();
  G__setgvp(G__PVOID);
  int result = loadLibraryForClass(c) ? 1:0;
  G__setgvp(varp);
  //NOTE: the check for the library is done since we can have a failure
  // if a CMS library has an incomplete set of Reflex dictionaries where 
  // the remaining dictionaries can be found by Cint.  If the library with
  // the Reflex dictionaries is loaded first, then the Cint library then any
  // requests for a Reflex::Type from the Reflex library will fail because for
  // some reason the loading of the Cint library causes Reflex to forget about
  // what types it already loaded from the Reflex library.  This problem was
  // seen for libDataFormatsMath and libMathCore.  I do not print an error message
  // since the dictionaries are actually loaded so things work fine.
  if(!result && 0 != strcmp(l,kDummyLibName) && gPrevious) {
    result = gPrevious(c,l);
  }
  return result;
}

static std::string 
classNameForRoot(const std::string& iCapName)
{
  //need to remove any 'std::' since ROOT ignores it
  static const boost::regex ex("std::");
  const std::string to("");
  
  return regex_replace(iCapName, ex, to, boost::match_default | boost::format_sed);
  
  return iCapName;
}

//Cint requires that we register the type and library containing the type
// before the autoloading will work
static
void registerTypes() {
  seal::PluginManager                       *db =  seal::PluginManager::get();
  seal::PluginManager::DirectoryIterator    dir;
  seal::ModuleCache::Iterator               plugin;
  seal::ModuleDescriptor                    *cache;
  unsigned                            i;
  const std::string mycat("Capability");

  //in order to determine if a name is from a class or a namespace, we will order
  // all the classes in descending order so that embedded classes will be seen before
  // their containing classes, that way we can say the containing class is a namespace
  // before finding out it is actually a class
  std::vector<std::string> classes;
  classes.reserve(1000);
  
  for (dir = db->beginDirectories(); dir != db->endDirectories(); ++dir) {
    for (plugin = (*dir)->begin(); plugin != (*dir)->end(); ++plugin) {
      for (cache=(*plugin)->cacheRoot(), i=0; i < cache->children(); ++i) {
        //std::cout <<" "<<cache->child(i)->token(0)<<std::endl;
        if (cache->child(i)->token(0) == mycat) {
          const std::string cap = cache->child(i)->token(1);
          //std::cout <<"  "<<cap<<std::endl;
          // check that cap starts with either LCGDict or LCGReflex (not really required)
          static const std::string cPrefix("LCGReflex/");
          if(cPrefix == cap.substr(0,cPrefix.size())) {
            std::string className = classNameForRoot(cap.c_str()+cPrefix.size());
            classes.push_back(className);
          }
        }
      }
    }
  }
  std::sort(classes.begin(), classes.end(), std::greater<std::string>());
  for(std::vector<std::string>::iterator itClass = classes.begin();
      itClass != classes.end();
      ++itClass) {
    
    const std::string& className = *itClass;
    //need to register namespaces and figure out if we have an embedded class
    static const std::string toFind(":<");
    std::string::size_type pos=0;
    while(std::string::npos != (pos = className.find_first_of(toFind,pos))) {
      if (className[pos] == '<') {break;}
      if (className.size() <= pos+1 || className[pos+1] != ':') {break;}
      //should check to see if this is a class or not
      G__set_class_autoloading_table(const_cast<char*>(className.substr(0,pos).c_str()),"");
      //std::cout <<"namespace "<<className.substr(0,pos).c_str()<<std::endl;
      pos += 2;
    }
    G__set_class_autoloading_table(const_cast<char*>(className.c_str()), const_cast<char*>(kDummyLibName));
    //std::cout <<"class "<<className.c_str()<<std::endl;
  }
}

//
// constructors and destructor
//
RootAutoLibraryLoader::RootAutoLibraryLoader() :
  classNameAttemptingToLoad_(0)
{
   seal::PluginManager::get()->initialise();
   gROOT->AddClassGenerator(this);
   ROOT::Cintex::Cintex::Enable();
   
   //std::cout <<"my loader"<<std::endl;
   //remember if the callback was already set so we can chain together our results
   gPrevious = G__p_class_autoloading;
   G__set_class_autoloading_callback(&ALL_AutoLoadCallback);
   registerTypes();
}


//
// member functions
//

TClass *
RootAutoLibraryLoader::GetClass(const char* classname, Bool_t load)
{
  if(classname == classNameAttemptingToLoad_) {
    std::cerr <<"WARNING: Reflex failed to create CINT dictionary for "<<classname<<std::endl;
    return 0;
  }
   TClass* returnValue = 0;
   //std::cout <<"looking for "<<classname <<" load "<<(load? "T":"F")<< std::endl;
   if (load) {
     //std::cout <<" going to call loadLibraryForClass"<<std::endl;
     if (loadLibraryForClass(classname)) {
       //use this to check for infinite recursion attempt
       classNameAttemptingToLoad_ = classname;       
       returnValue = gROOT->GetClass(classname,kFALSE);
       classNameAttemptingToLoad_ = 0;
     }
   }
   return returnValue;
}


TClass *
RootAutoLibraryLoader::GetClass(const type_info& typeinfo, Bool_t load)
{
  //std::cout <<"looking for type "<<typeinfo.name()<<std::endl;
   TClass* returnValue = 0;
   if(load){
      return GetClass(typeinfo.name(), load);
   }
   return returnValue;
}

void
RootAutoLibraryLoader::enable()
{
   //static BareRootProductGetter s_getter;
   //static edm::EDProductGetter::Operate s_op(&s_getter);
   static RootAutoLibraryLoader s_loader;
}




void
RootAutoLibraryLoader::loadAll()
{
  // std::cout <<"LoadAllDictionaries"<<std::endl;
  enable();
  
  seal::PluginManager                       *db =  seal::PluginManager::get();
  seal::PluginManager::DirectoryIterator    dir;
  seal::ModuleCache::Iterator               plugin;
  seal::ModuleDescriptor                    *cache;
  unsigned                            i;
  
  const std::string mycat("Capability");
  
  for (dir = db->beginDirectories(); dir != db->endDirectories(); ++dir) {
    for (plugin = (*dir)->begin(); plugin != (*dir)->end(); ++plugin) {
      for (cache=(*plugin)->cacheRoot(), i=0; i < cache->children(); ++i) {
        //std::cout <<" "<<cache->child(i)->token(0)<<std::endl;
        if (cache->child(i)->token(0) == mycat) {
          const std::string cap = cache->child(i)->token(1);
          //std::cout <<"  "<<cap<<std::endl;
          // check that cap starts with either LCGDict or LCGReflex (not really required)
          static const std::string cPrefix("LCGReflex/");
          if(cPrefix == cap.substr(0,cPrefix.size())) {
            seal::PluginCapabilities::get()->load(cap);
          }
          break;
        }
      }
    }
  }
}
}

//ClassImp(RootAutoLibraryLoader)
