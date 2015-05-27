// -*- C++ -*-
//
// Package:     LibraryLoader
// Class  :     FWLiteEnabler
// 
// Implementation:
//     <Notes on implementation>
//
//

// system include files

// user include files
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "FWCore/FWLite/interface/Enable.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//


//
// constructors and destructor
//
// Note: this ctor will never be invoked.
// This class is simply a wrapper for the static function fwlite::enable,
// which cannot be used directly in ROOT macros.
FWLiteEnabler::FWLiteEnabler() {
}


//
// member functions
//

void
FWLiteEnabler::enable() {
  fwlite::enable();
}

