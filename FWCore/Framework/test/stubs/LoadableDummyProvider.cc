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
// $Id: LoadableDummyProvider.cc,v 1.1 2005/05/29 02:29:54 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/CoreFramework/test/DummyProxyProvider.h"

#include "FWCore/CoreFramework/test/DummyFinder.h"
#include "FWCore/CoreFramework/interface/ModuleFactory.h"

namespace edm {
   class ParameterSet;
}
class LoadableDummyProvider : public edm::eventsetup::test::DummyProxyProvider {
public:
   LoadableDummyProvider(const edm::ParameterSet&) {}
};

DEFINE_FWK_EVENTSETUP_MODULE(LoadableDummyProvider)
