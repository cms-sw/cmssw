// -*- C++ -*-
//
// Package:     test
// Class  :     LoadableDummyESSource
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu May 26 13:48:03 EDT 2005
// $Id: LoadableDummyESSource.cc,v 1.2 2005/12/22 20:28:28 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/test/DummyProxyProvider.h"

#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"


namespace edm {
   class ParameterSet;
}

class LoadableDummyESSource : public edm::eventsetup::test::DummyProxyProvider, public DummyFinder
{
public:
   LoadableDummyESSource(const edm::ParameterSet& iPSet)
   : DummyProxyProvider( edm::eventsetup::test::DummyData(iPSet.getUntrackedParameter<int>("value",2))){
      setInterval(edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),
                                        edm::IOVSyncValue::endOfTime()));
   }
};

DEFINE_FWK_EVENTSETUP_SOURCE(LoadableDummyESSource);
