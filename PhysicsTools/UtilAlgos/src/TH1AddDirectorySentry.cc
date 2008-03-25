// -*- C++ -*-
//
// Package:     UtilAlgos
// Class  :     TH1AddDirectorySentry
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Nov  8 12:16:02 EST 2007
// $Id$
//

// system include files
#include "TH1.h"

// user include files
#include "PhysicsTools/UtilAlgos/interface/TH1AddDirectorySentry.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TH1AddDirectorySentry::TH1AddDirectorySentry():
   status_(TH1::AddDirectoryStatus())
{
   TH1::AddDirectory(true);
}

TH1AddDirectorySentry::~TH1AddDirectorySentry()
{
   TH1::AddDirectory(status_);
}

