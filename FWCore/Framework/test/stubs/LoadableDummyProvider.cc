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
//

// system include files

// user include files
#include "FWCore/Framework/test/DummyESProductResolverProvider.h"

#include "FWCore/Framework/interface/ModuleFactory.h"

namespace edm {
  class ParameterSet;
}
class LoadableDummyProvider : public edm::eventsetup::test::DummyESProductResolverProvider {
public:
  LoadableDummyProvider(const edm::ParameterSet& iPSet)
      : DummyESProductResolverProvider(edm::eventsetup::test::DummyData(iPSet.getUntrackedParameter<int>("value", 1))) {
  }
};

DEFINE_FWK_EVENTSETUP_MODULE(LoadableDummyProvider);
