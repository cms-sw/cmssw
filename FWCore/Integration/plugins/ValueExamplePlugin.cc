// -*- C++ -*-
//
// Package:     test
// Class  :     ValueExample
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:52:01 EDT 2005
//

// system include files

// user include files
#include "ValueExample.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
using namespace edm::serviceregistry;
DEFINE_FWK_SERVICE_MAKER(ValueExample, ParameterSetMaker<ValueExample>);
