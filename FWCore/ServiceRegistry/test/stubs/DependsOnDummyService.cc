// -*- C++ -*-
//
// Package:     test
// Class  :     DependsOnDummyService
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:52:01 EDT 2005
// $Id: DependsOnDummyService.cc,v 1.2 2005/09/10 02:08:49 wmtan Exp $
//

#include "FWCore/ServiceRegistry/test/stubs/DependsOnDummyService.h"
#include "FWCore/ServiceRegistry/test/stubs/DummyService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

using namespace testserviceregistry;

DependsOnDummyService::DependsOnDummyService():
value_(edm::Service<DummyService>()->value())
{
}

DependsOnDummyService::~DependsOnDummyService()
{
}

void DependsOnDummyService::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}
