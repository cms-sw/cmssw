// -*- C++ -*-
//
// Package:     test
// Class  :     DummyServiceE0
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  David Dagenhart
// $Id: DummyServiceE0.cc,v 1.2 2007/08/06 20:54:12 wmtan Exp $

#include "FWCore/ServiceRegistry/test/stubs/DummyServiceE0.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <iostream>

using namespace testserviceregistry;

namespace {
  int testCounter = 0;
}

// ------------------------------------------------------

void DummyServiceBase::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DummyServiceE0::DummyServiceE0(const edm::ParameterSet& iPSet,
                             edm::ActivityRegistry&iAR)
{
  std::cout << "DummyServiceE0 Constructor " << testCounter << std::endl;
  if (testCounter != 0) abort();
  ++testCounter;

  iAR.watchPostBeginJob(this, &testserviceregistry::DummyServiceE0::postBeginJob);
  iAR.watchPostEndJob(this, &testserviceregistry::DummyServiceE0::postEndJob);
}

void DummyServiceE0::postBeginJob() {
  std::cout << "DummyServiceE0 PostBeginJob " << testCounter << std::endl;
  if (testCounter != 5) abort();
  ++testCounter;
}

void DummyServiceE0::postEndJob() {
  std::cout << "DummyServiceE0 PostEndJob " << testCounter << std::endl;
  if (testCounter != 14) abort();
  ++testCounter;
}

DummyServiceE0::~DummyServiceE0()
{
  std::cout << "DummyServiceE0 Destructor " << testCounter << std::endl;
  if (testCounter != 19) abort();
  ++testCounter;
}

// ------------------------------------------------------

DummyServiceA1::DummyServiceA1(const edm::ParameterSet& iPSet,
                             edm::ActivityRegistry&iAR)
{
  std::cout << "DummyServiceA1 Constructor " << testCounter << std::endl;
  if (testCounter != 1) abort();
  ++testCounter;

  iAR.watchPostBeginJob(this, &testserviceregistry::DummyServiceA1::postBeginJob);
  iAR.watchPostEndJob(this, &testserviceregistry::DummyServiceA1::postEndJob);
}

void DummyServiceA1::postBeginJob() {
  std::cout << "DummyServiceA1 PostBeginJob " << testCounter << std::endl;
  if (testCounter != 6) abort();
  ++testCounter;
}

void DummyServiceA1::postEndJob() {
  std::cout << "DummyServiceA1 PostEndJob " << testCounter << std::endl;
  if (testCounter != 13) abort();
  ++testCounter;
}

DummyServiceA1::~DummyServiceA1()
{
  std::cout << "DummyServiceA1 Destructor " << testCounter << std::endl;
  if (testCounter != 18) abort();
  ++testCounter;
}

// ------------------------------------------------------

DummyServiceD2::DummyServiceD2(const edm::ParameterSet& iPSet,
                             edm::ActivityRegistry&iAR)
{
  std::cout << "DummyServiceD2 Constructor " << testCounter << std::endl;
  if (testCounter != 2) abort();
  ++testCounter;

  iAR.watchPostBeginJob(this, &testserviceregistry::DummyServiceD2::postBeginJob);
  iAR.watchPostEndJob(this, &testserviceregistry::DummyServiceD2::postEndJob);
}

void DummyServiceD2::postBeginJob() {
  std::cout << "DummyServiceD2 PostBeginJob " << testCounter << std::endl;
  if (testCounter != 7) abort();
  ++testCounter;
}

void DummyServiceD2::postEndJob() {
  std::cout << "DummyServiceD2 PostEndJob " << testCounter << std::endl;
  if (testCounter != 12) abort();
  ++testCounter;
}

DummyServiceD2::~DummyServiceD2()
{
  std::cout << "DummyServiceD2 Destructor " << testCounter << std::endl;
  if (testCounter != 17) abort();
  ++testCounter;
}

// ------------------------------------------------------

DummyServiceB3::DummyServiceB3(const edm::ParameterSet& iPSet,
                             edm::ActivityRegistry&iAR)
{
  // Make this service dependent on service D2 in order to "On Demand Creation"
  edm::Service<DummyServiceD2>().isAvailable();

  std::cout << "DummyServiceB3 Constructor " << testCounter << std::endl;
  if (testCounter != 3) abort();
  ++testCounter;

  iAR.watchPostBeginJob(this, &testserviceregistry::DummyServiceB3::postBeginJob);
  iAR.watchPostEndJob(this, &testserviceregistry::DummyServiceB3::postEndJob);
}

void DummyServiceB3::postBeginJob() {
  std::cout << "DummyServiceB3 PostBeginJob " << testCounter << std::endl;
  if (testCounter != 8) abort();
  ++testCounter;
}

void DummyServiceB3::postEndJob() {
  std::cout << "DummyServiceB3 PostEndJob " << testCounter << std::endl;
  if (testCounter != 11) abort();
  ++testCounter;
}

DummyServiceB3::~DummyServiceB3()
{
  std::cout << "DummyServiceB3 Destructor " << testCounter << std::endl;
  if (testCounter != 16) abort();
  ++testCounter;
}

// ------------------------------------------------------

DummyServiceC4::DummyServiceC4(const edm::ParameterSet& iPSet,
                             edm::ActivityRegistry&iAR)
{
  std::cout << "DummyServiceC4 Constructor " << testCounter << std::endl;
  if (testCounter != 4) abort();
  ++testCounter;

  iAR.watchPostBeginJob(this, &testserviceregistry::DummyServiceC4::postBeginJob);
  iAR.watchPostEndJob(this, &testserviceregistry::DummyServiceC4::postEndJob);
}

void DummyServiceC4::postBeginJob() {
  std::cout << "DummyServiceC4 PostBeginJob " << testCounter << std::endl;
  if (testCounter != 9) abort();
  ++testCounter;
}

void DummyServiceC4::postEndJob() {
  std::cout << "DummyServiceC4 PostEndJob " << testCounter << std::endl;
  if (testCounter != 10) abort();
  ++testCounter;
}

DummyServiceC4::~DummyServiceC4()
{
  std::cout << "DummyServiceC4 Destructor " << testCounter << std::endl;
  if (testCounter != 15) abort();
  ++testCounter;
}
