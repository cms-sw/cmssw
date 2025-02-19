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
// $Id: LoadableDummyESSource.cc,v 1.6 2009/09/23 23:31:10 wmtan Exp $
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
     unsigned int startRun = iPSet.getUntrackedParameter<unsigned int>("startRunForIOV",0);
     if ( 0 == startRun ) {
       setInterval(edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),
                                         edm::IOVSyncValue::endOfTime()));
     } else {
       setInterval(edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(startRun,0,0)),
                                         edm::IOVSyncValue::endOfTime()));
       
     }
   }
};

DEFINE_FWK_EVENTSETUP_SOURCE(LoadableDummyESSource);
