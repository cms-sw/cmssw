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
//

// system include files

#include <iostream>
// user include files
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//


//
// constructors and destructor
//
AutoLibraryLoader::AutoLibraryLoader() {
}


//
// member functions
//

void
AutoLibraryLoader::enable() {
  std::cerr << "WARNING: AutoLibraryloader::enable() and AutoLibraryLoader.h are deprecated.\n" <<
  "Use FWLiteEnabler::enable() and FWLiteEnabler.h instead"  << std::endl;
  FWLiteEnabler::enable();
}

void
AutoLibraryLoader::loadAll()
{
  std::cerr << "WARNING: AutoLibraryloader::loadAll() and AutoLibraryLoader.h are deprecated.\n" <<
  "Use FWLiteEnabler::enable() and FWLiteEnabler.h instead"  << std::endl;
  FWLiteEnabler::enable();
}

