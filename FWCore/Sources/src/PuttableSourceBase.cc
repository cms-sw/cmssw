// -*- C++ -*-
//
// Package:     FWCore/Sources
// Class  :     PuttableSourceBase
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  root
//         Created:  Tue, 26 Sep 2017 20:52:26 GMT
//

// system include files

// user include files
#include "FWCore/Sources/interface/PuttableSourceBase.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ExceptionHelpers.h"

using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PuttableSourceBase::PuttableSourceBase(ParameterSet const& iPSet, InputSourceDescription const& iISD)
    : InputSource(iPSet, iISD) {}

void PuttableSourceBase::registerProducts() { registerProducts(this, &productRegistryUpdate(), moduleDescription()); }

void PuttableSourceBase::beginJob() {
  auto r = productRegistry();
  auto const runLookup = r->productLookup(InRun);
  auto const lumiLookup = r->productLookup(InLumi);
  auto const eventLookup = r->productLookup(InEvent);
  auto const& processName = moduleDescription().processName();
  auto const& moduleLabel = moduleDescription().moduleLabel();

  auto const& runModuleToIndicies = runLookup->indiciesForModulesInProcess(processName);
  auto const& lumiModuleToIndicies = lumiLookup->indiciesForModulesInProcess(processName);
  auto const& eventModuleToIndicies = eventLookup->indiciesForModulesInProcess(processName);
  resolvePutIndicies(InRun, runModuleToIndicies, moduleLabel);
  resolvePutIndicies(InLumi, lumiModuleToIndicies, moduleLabel);
  resolvePutIndicies(InEvent, eventModuleToIndicies, moduleLabel);
}

void PuttableSourceBase::doBeginRun(RunPrincipal& rp, ProcessContext const*) {
  Run run(rp, moduleDescription(), nullptr, false);
  run.setProducer(this);
  callWithTryCatchAndPrint<void>([this, &run]() { beginRun(run); }, "Calling Source::beginRun");
  commit_(run);
}

void PuttableSourceBase::doBeginLumi(LuminosityBlockPrincipal& lbp, ProcessContext const*) {
  LuminosityBlock lb(lbp, moduleDescription(), nullptr, false);
  lb.setProducer(this);
  callWithTryCatchAndPrint<void>([this, &lb]() { beginLuminosityBlock(lb); }, "Calling Source::beginLuminosityBlock");
  commit_(lb);
}

void PuttableSourceBase::beginRun(Run&) {}

void PuttableSourceBase::beginLuminosityBlock(LuminosityBlock&) {}
