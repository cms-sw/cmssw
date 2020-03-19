#include <string>

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "HLTrigger/Timer/interface/EDMModuleType.h"

namespace edm {

  EDMModuleType edmModuleTypeEnum(edm::ModuleDescription const& module) {
    auto const& registry = *edm::pset::Registry::instance();
    auto const& pset = *registry.getMapped(module.parameterSetID());

    if (not pset.existsAs<std::string>("@module_edm_type"))
      return EDMModuleType::kUnknown;

    std::string const& t = pset.getParameter<std::string>("@module_edm_type");
    for (EDMModuleType v : {EDMModuleType::kSource,
                            EDMModuleType::kESSource,
                            EDMModuleType::kESProducer,
                            EDMModuleType::kEDAnalyzer,
                            EDMModuleType::kEDProducer,
                            EDMModuleType::kEDFilter,
                            EDMModuleType::kOutputModule}) {
      if (t == module_type_desc[static_cast<std::underlying_type_t<EDMModuleType>>(v)])
        return v;
    }
    return EDMModuleType::kUnknown;
  }

  const char* edmModuleType(edm::ModuleDescription const& module) {
    return module_type_desc[static_cast<std::underlying_type_t<EDMModuleType>>(edmModuleTypeEnum(module))];
  }

}  // namespace edm
