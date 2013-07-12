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
// $Id: TH1AddDirectorySentry.cc,v 1.1 2009/03/03 13:07:30 llista Exp $
//

// system include files
#include "TH1.h"

// user include files
#include "CommonTools/Utils/interface/TH1AddDirectorySentry.h"


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

