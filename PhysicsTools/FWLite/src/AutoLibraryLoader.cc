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
// $Id$
//

// system include files
#include <iostream>
#include "TROOT.h"

// user include files
#include "PhysicsTools/FWLite/src/AutoLibraryLoader.h"

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

//
// constructors and destructor
//
AutoLibraryLoader::AutoLibraryLoader()
{
   seal::PluginManager::get()->initialise();
   gROOT->AddClassGenerator(this);
   seal::cintex::Cintex::enable();
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
      static const std::string cPrefix("LCGReflex/");
      //std::cout <<"asking to find "<<cPrefix+classname<<std::endl;
      seal::PluginCapabilities::get()->load(cPrefix+classname);

      seal::reflex::Type t = seal::reflex::Type::byName(classname);
      if(seal::reflex::Type() != t ) {
	 //std::cout <<"loaded "<<classname<<std::endl;
	 return gROOT->GetClass(classname,kFALSE);
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
   static AutoLibraryLoader s_loader;
}

ClassImp(AutoLibraryLoader);
