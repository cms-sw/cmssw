// -*- C++ -*-
//
// Package:     test
// Class  :     DummyFinder
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu May 26 10:52:25 EDT 2005
//

// system include files

// user include files
#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"

namespace edm {
   class ParameterSet;
}
class LoadableDummyFinder : public DummyFinder {
public:
   LoadableDummyFinder(const edm::ParameterSet&) {}
};

DEFINE_FWK_EVENTSETUP_SOURCE(LoadableDummyFinder);
