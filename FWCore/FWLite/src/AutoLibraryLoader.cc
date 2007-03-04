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
// $Id: AutoLibraryLoader.cc,v 1.15 2007/02/01 16:10:20 wmtan Exp $
//

// system include files
#include <iostream>
#include "TROOT.h"
#include "G__ci.h"
#include "boost/regex.hpp"

// user include files
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/FWLite/src/BareRootProductGetter.h"

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
