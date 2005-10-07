// -*- C++ -*-
//
// Package:     Framework
// Class  :     SignallingProductRegistry
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Sep 23 16:52:50 CEST 2005
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/src/SignallingProductRegistry.h"

using namespace edm;
//
// member functions
//

void SignallingProductRegistry::addCalled(BranchDescription const& iProd)
{
   productAddedSignal_(iProd);
}
