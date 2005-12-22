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
// $Id: LoadableDummyProvider.cc,v 1.3 2005/07/14 22:50:53 wmtan Exp $
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
   LoadableDummyProvider(const edm::ParameterSet& iPSet) 
   :DummyProxyProvider( edm::eventsetup::test::DummyData(iPSet.getUntrackedParameter<int>("value",1))) {}
};

DEFINE_FWK_EVENTSETUP_MODULE(LoadableDummyProvider)
