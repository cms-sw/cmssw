// -*- C++ -*-
//
// Package:     test
// Class  :     LoadableDummyEventSetupRecordRetriever
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu May 26 13:48:03 EDT 2005
//

// user include files

// system include files

#include "FWCore/Framework/test/DummyEventSetupRecordRetriever.h"
#include "FWCore/Framework/interface/SourceFactory.h"

namespace edm {
   class ParameterSet;
}

class LoadableDummyEventSetupRecordRetriever : public edm::DummyEventSetupRecordRetriever {
public:
  LoadableDummyEventSetupRecordRetriever(edm::ParameterSet const&) {
   }
};

DEFINE_FWK_EVENTSETUP_SOURCE(LoadableDummyEventSetupRecordRetriever);
