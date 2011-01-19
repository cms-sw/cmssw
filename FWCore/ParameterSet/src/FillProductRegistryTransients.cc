// -*- C++ -*-
//
// Package:     ParameterSet
//
// fillProductRegistry()
// This free function reads information from the process parameter set
// and writes information derived from this into the ProductRegistry.
// It really does not belong in ParameterSet, but ParameterSet is the only existing
// package in which it can go without introducing additional package dependencies.

#include "FWCore/ParameterSet/interface/FillProductRegistryTransients.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

namespace edm {
  void
  fillProductRegistryTransients(ProcessConfiguration const& pc, ProductRegistry const& preg, bool okToRegister) {
    std::string const triggerResults = std::string("TriggerResults");
    std::string const triggerResultsInserter = std::string("TriggerResultsInserter");
    std::string const triggerPaths = std::string("@trigger_paths");
    std::string const source = std::string("source");
    std::string const input = std::string("@main_input");
    ProcessConfigurationID pcid = pc.id();
    std::string const& processName = pc.processName();
    ParameterSetID const& processParameterSetID = pc.parameterSetID();
    ParameterSet const* processParameterSet = pset::Registry::instance()->getMapped(processParameterSetID);
    if(0 == processParameterSet || processParameterSet->empty()) {
      return;
    }
    for(ProductRegistry::ProductList::const_iterator it = preg.productList().begin(),
        itEnd = preg.productList().end();
        it != itEnd; ++it) {
      BranchDescription const& bd = it->second;
      if(processName != bd.processName()) {
        continue;
      }
      std::string moduleLabel = bd.moduleLabel();
      bool isTriggerResults = (moduleLabel == triggerResults);
      if(isTriggerResults) {
        moduleLabel = triggerPaths;
      } else if(moduleLabel == source) {
        moduleLabel = input;
      }
      if(processParameterSet->existsAs<ParameterSet>(moduleLabel)) {
        ParameterSet const& moduleParameterSet = processParameterSet->getParameterSet(moduleLabel);
        if(okToRegister && !moduleParameterSet.isRegistered()) {
          ParameterSet moduleParameterSetCopy = processParameterSet->getParameter<ParameterSet>(moduleLabel);
          moduleParameterSetCopy.registerIt();
          bd.parameterSetIDs().insert(std::make_pair(pcid, moduleParameterSetCopy.id()));
        } else {
          bd.parameterSetIDs().insert(std::make_pair(pcid, moduleParameterSet.id()));
        }
        if(isTriggerResults) {
          bd.moduleNames().insert(std::make_pair(pcid, triggerResultsInserter));
        } else {
          bd.moduleNames().insert(std::make_pair(pcid, moduleParameterSet.getParameter<std::string>("@module_type")));
        }
      }
    }
  }

  void
  fillProductRegistryTransients(std::vector<ProcessConfiguration> const& pcVec, ProductRegistry const& preg, bool okToRegister) {
    typedef std::vector<ProcessConfiguration>::const_iterator PCIter;
    for(PCIter i = pcVec.begin(), iEnd = pcVec.end(); i != iEnd; ++i) {
      fillProductRegistryTransients(*i, preg, okToRegister);
    }
  }
}
