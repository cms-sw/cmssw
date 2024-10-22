#ifndef HLTrigger_Timer_interface_EDMModuleType_h
#define HLTrigger_Timer_interface_EDMModuleType_h

#include "DataFormats/Provenance/interface/ModuleDescription.h"

namespace edm {

  enum class EDMModuleType {
    kUnknown,
    kSource,
    kESSource,
    kESProducer,
    kEDAnalyzer,
    kEDProducer,
    kEDFilter,
    kOutputModule
  };

  constexpr const char* module_type_desc[]{
      "Unknown", "Source", "ESSource", "ESProducer", "EDAnalyzer", "EDProducer", "EDFilter", "OutputModule"};

  EDMModuleType edmModuleTypeEnum(edm::ModuleDescription const& module);

  const char* edmModuleType(edm::ModuleDescription const& module);

}  // namespace edm

#endif  // not defined HLTrigger_Timer_interface_EDMModuleType_h
