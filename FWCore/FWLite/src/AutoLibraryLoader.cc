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
// $Id: AutoLibraryLoader.cc,v 1.16 2007/03/04 05:25:01 wmtan Exp $
//

// system include files
#include <iostream>
#include "TROOT.h"

// user include files
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/FWLite/src/BareRootProductGetter.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
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
}


//
// member functions
//

void
AutoLibraryLoader::enable()
{
   edmplugin::PluginManager::configure(edmplugin::standard::config());
   static BareRootProductGetter s_getter;
   static edm::EDProductGetter::Operate s_op(&s_getter);
   edm::RootAutoLibraryLoader::enable();
}




void
AutoLibraryLoader::loadAll()
{
  // std::cout <<"LoadAllDictionaries"<<std::endl;
  enable();
  edm::RootAutoLibraryLoader::loadAll();
}


ClassImp(AutoLibraryLoader)
