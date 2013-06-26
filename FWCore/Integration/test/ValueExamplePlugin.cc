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
// $Id: ValueExamplePlugin.cc,v 1.1 2007/04/09 23:04:28 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Integration/test/ValueExample.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
using namespace edm::serviceregistry;
DEFINE_FWK_SERVICE_MAKER(ValueExample,ParameterSetMaker<ValueExample>);
