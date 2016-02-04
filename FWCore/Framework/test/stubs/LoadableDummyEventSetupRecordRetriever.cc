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
// $Id: LoadableDummyEventSetupRecordRetriever.cc,v 1.3 2007/08/07 22:34:19 wmtan Exp $
//

// system include files

// user include files

#include "FWCore/Framework/test/DummyEventSetupRecordRetriever.h"
#include "FWCore/Framework/interface/SourceFactory.h"


namespace edm {
   class ParameterSet;
}

class LoadableDummyEventSetupRecordRetriever : public edm::DummyEventSetupRecordRetriever
{
public:
  LoadableDummyEventSetupRecordRetriever(const edm::ParameterSet& iPSet) {
   }
};

DEFINE_FWK_EVENTSETUP_SOURCE(LoadableDummyEventSetupRecordRetriever);
