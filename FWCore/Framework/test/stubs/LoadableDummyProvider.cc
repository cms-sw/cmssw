// -*- C++ -*-
//
// Package:     test
// Class  :     LoadableDummyProvider
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu May 26 13:48:03 EDT 2005
// $Id: LoadableDummyProvider.cc,v 1.2 2005/06/23 20:01:12 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Framework/test/DummyProxyProvider.h"

#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

namespace edm {
   class ParameterSet;
}
class LoadableDummyProvider : public edm::eventsetup::test::DummyProxyProvider {
public:
   LoadableDummyProvider(const edm::ParameterSet&) {}
};

DEFINE_FWK_EVENTSETUP_MODULE(LoadableDummyProvider)
