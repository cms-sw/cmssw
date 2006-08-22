// -*- C++ -*-
//
// Package:     LibraryLoader
// Class  :     AutoLibraryLoader
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed Nov 30 14:55:01 EST 2005
// $Id: AutoLibraryLoader.cc,v 1.3 2006/08/08 23:57:33 chrjones Exp $
//

// system include files
#include <iostream>
#include "TROOT.h"
#include "G__ci.h"
#include "common.h"
#include "boost/regex.hpp"

// user include files
#include "FWCore/FWLite/src/AutoLibraryLoader.h"
#include "FWCore/FWLite/src/stdNamespaceAdder.h"
#include "FWCore/FWLite/src/BareRootProductGetter.h"

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

//This is actually defined within ROOT's v6_struct.cxx file but is not declared static
// I want to use it so that if the autoloading is already turned on, I can call the previously declared routine
extern CallbackPtr G__p_class_autoloading;

static 
bool loadLibraryForClass( const char* classname )
{  
  //std::cout <<"asking to find "<<classname<<std::endl;
  static const std::string cPrefix("LCGReflex/");
  //std::cout <<"asking to find "<<cPrefix+classname<<std::endl;
  seal::PluginCapabilities::get()->load(cPrefix+classname);
  
  ROOT::Reflex::Type t = ROOT::Reflex::Type::ByName(classname);
  if(ROOT::Reflex::Type() != t ) {
    //std::cout <<"loaded "<<classname<<std::endl;
    return true;
  } 
  //see if adding a std namespace helps
  std::string name = fwlite::stdNamespaceAdder(classname);
  
  seal::PluginCapabilities::get()->load(cPrefix+name);
  
  t = ROOT::Reflex::Type::ByName(classname);
  return ROOT::Reflex::Type() != t;
}

//Based on code in ROOT's TCint.cxx file

/* extern "C" */ static int ALL_AutoLoadCallback(char *c, char *l) {
  ULong_t varp = G__getgvp();
  G__setgvp(G__PVOID);
  int result = loadLibraryForClass(c) ? 1:0;
  G__setgvp(varp);
  if(!result && gPrevious) {
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
            std::string className = classNameForRoot( cap.c_str()+cPrefix.size() );
            //need to register namespaces and figure out if we have an embedded class
            static const std::string toFind(":<");
            std::string::size_type pos=0;
            while(std::string::npos != (pos = className.find_first_of(toFind,pos)) ) {
              if( className[pos] == '<') {break;}
              if (className.size() <= pos+1 or className[pos+1] != ':') {break;}
              //should check to see if this is a class or not
              G__set_class_autoloading_table(const_cast<char*>( className.substr(0,pos).c_str() ),"");
              //std::cout <<"namespace "<<className.substr(0,pos).c_str()<<std::endl;
              pos +=2;
            }
            G__set_class_autoloading_table(const_cast<char*>( className.c_str()),"dummy");
            //std::cout <<"class "<<className.c_str()<<std::endl;
          }
        }
      }
    }
  }
}

//
// constructors and destructor
//
AutoLibraryLoader::AutoLibraryLoader()
{
   seal::PluginManager::get()->initialise();
   gROOT->AddClassGenerator(this);
   ROOT::Cintex::Cintex::Enable();
   
   //remember if the callback was already set so we can chain together our results
   gPrevious = G__p_class_autoloading;
   G__set_class_autoloading_callback(&ALL_AutoLoadCallback);
   registerTypes();
}


//
// member functions
//

TClass *
AutoLibraryLoader::GetClass(const char* classname, Bool_t load)
{
   TClass* returnValue = 0;
//   std::cout <<"looking for "<<classname <<" load "<<(load? "T":"F")<< std::endl;
   if(load) {
     if(loadLibraryForClass(classname) ) {
       returnValue = gROOT->GetClass(classname,kFALSE);
     }
   }
   return returnValue;
}


TClass *
AutoLibraryLoader::GetClass(const type_info& typeinfo, Bool_t load)
{
   //std::cout <<"looking for type "<<typeinfo.name()<<std::endl;
   TClass* returnValue = 0;
   if(load){
      return GetClass(typeinfo.name(), load);
   }
   return returnValue;
}

void
AutoLibraryLoader::enable()
{
   static BareRootProductGetter s_getter;
   static edm::EDProductGetter::Operate s_op(&s_getter);
   static AutoLibraryLoader s_loader;
}




void
AutoLibraryLoader::loadAll()
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


ClassImp(AutoLibraryLoader);
