// -*- C++ -*-
//
// Package:     Framework
// 
#include "FWCore/Framework/interface/FillProductRegistryTransients.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

namespace edm {
  void
  fillProductRegistryTransients(std::vector<ProcessConfiguration> const& pcVec, ProductRegistry& preg) {
    std::string const triggerResults = std::string("TriggerResults");
    std::string const triggerResultsInserter = std::string("TriggerResultsInserter");
    std::string const triggerPaths = std::string("@trigger_paths");
    std::string const source = std::string("source");
    std::string const input = std::string("@main_input");
    typedef std::vector<ProcessConfiguration>::const_iterator PCIter;
    for (PCIter i = pcVec.begin(), iEnd = pcVec.end(); i != iEnd; ++i) {
      ProcessConfigurationID pcid = i->id();
      std::string const& processName = i->processName();
      ParameterSetID const& processParameterSetID = i->parameterSetID();
      ParameterSet processParameterSet;
      pset::Registry::instance()->getMapped(processParameterSetID, processParameterSet);
      if (processParameterSet.empty()) {
        continue;
      }
      for (ProductRegistry::ProductList::const_iterator it = preg.productList().begin(),
          itEnd = preg.productList().end();
          it != itEnd; ++it) {
        BranchDescription const& bd = it->second;
        if (processName != bd.processName()) {
          continue;
        }
        std::string moduleLabel = bd.moduleLabel();
        bool isTriggerResults = (moduleLabel == triggerResults);
        if (isTriggerResults) {
          moduleLabel = triggerPaths;
        } else if (moduleLabel == source) {
          moduleLabel = input;
        } 
	if (processParameterSet.existsAs<ParameterSet>(moduleLabel)) {
          ParameterSet const& moduleParameterSet = processParameterSet.getParameterSet(moduleLabel);
          bd.parameterSetIDs().insert(std::make_pair(pcid, moduleParameterSet.id()));
	  if (isTriggerResults) {
            bd.moduleNames().insert(std::make_pair(pcid, triggerResultsInserter));
          } else {
            bd.moduleNames().insert(std::make_pair(pcid, moduleParameterSet.getParameter<std::string>("@module_type")));
	  }
        }
      }
    }
  }
}
