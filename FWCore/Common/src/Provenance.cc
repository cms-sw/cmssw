#include "FWCore/Common/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

namespace edm {
  static std::string const triggerResults("TriggerResults");
  static std::string const source("source");
  static std::string const triggerResultsInserter("TriggerResultsInserter");

  ParameterSet const& parameterSet(Provenance const& provenance) {
    ProcessConfiguration pc;
    provenance.getProcessConfiguration(pc);
    ParameterSet const& processParameterSet = *pset::Registry::instance()->getMapped(pc.parameterSetID());
    std::string const& label = provenance.moduleLabel();
    if (!processParameterSet.existsAs<ParameterSet>(label)) {
      // Trigger results and input sources are special cases
      if(label == triggerResults) {
        return processParameterSet.getParameterSet("@trigger_paths");
      } else if(label == source) {
        return processParameterSet.getParameterSet("@main_input");
      }
    }
    return processParameterSet.getParameterSet(label);
  }

  std::string moduleName(Provenance const& provenance) {
    // Trigger results ia a special case
    if(provenance.moduleLabel() == triggerResults) {
      return triggerResultsInserter; 
    }
    return parameterSet(provenance).getParameter<std::string>("@module_type");
  }
}
