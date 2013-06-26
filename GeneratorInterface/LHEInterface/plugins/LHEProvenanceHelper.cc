#include <string>
#include <vector>
#include "GeneratorInterface/LHEInterface/plugins/LHEProvenanceHelper.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

#include "FWCore/Utilities/interface/GetPassID.h"

#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

namespace edm {
  LHEProvenanceHelper::LHEProvenanceHelper(TypeID const& eventProductType, TypeID const& runProductType)
        : eventProductBranchDescription_(BranchDescription(
                                                  InEvent
                                                  , "source"
                                                  , "LHEFile"
                                                  // , "LHE"
                                                  , "LHEEventProduct"
                                                  , "LHEEventProduct"
                                                  , ""
                                                  , "LHESource"
                                                  , ParameterSetID()
                                                  , TypeWithDict(eventProductType.typeInfo())
                                                  , false))
        , runProductBranchDescription_(BranchDescription(
                                                  InRun
                                                  , "source"
                                                  , "LHEFile"
                                                  // , "LHE"
                                                  , "LHERunInfoProduct"
                                                  , "LHERunInfoProduct"
                                                  , ""
                                                  , "LHESource"
                                                  , ParameterSetID()
                                                  , TypeWithDict(runProductType.typeInfo())
                                                  , false))
        , eventProductProvenance_(eventProductBranchDescription_.branchID())
        , processParameterSet_() {
    
    // Now we create a process parameter set for the "LHC" process.
    // We don't currently use the untracked parameters, However, we make them available, just in case.
    std::string const& moduleLabel = eventProductBranchDescription_.moduleLabel();
    std::string const& processName = eventProductBranchDescription_.processName();
    std::string const& moduleName = eventProductBranchDescription_.moduleName();
    typedef std::vector<std::string> vstring;
    vstring empty;

    vstring modlbl;
    modlbl.reserve(1);
    modlbl.push_back(moduleLabel);
    processParameterSet_.addParameter("@all_sources", modlbl);

    ParameterSet triggerPaths;
    triggerPaths.addParameter<vstring>("@trigger_paths", empty);
    processParameterSet_.addParameter<ParameterSet>("@trigger_paths", triggerPaths);

    ParameterSet pseudoInput;
    pseudoInput.addParameter<std::string>("@module_edm_type", "Source");
    pseudoInput.addParameter<std::string>("@module_label", moduleLabel);
    pseudoInput.addParameter<std::string>("@module_type", moduleName);
    processParameterSet_.addParameter<ParameterSet>(moduleLabel, pseudoInput);

    processParameterSet_.addParameter<vstring>("@all_esmodules", empty);
    processParameterSet_.addParameter<vstring>("@all_esprefers", empty);
    processParameterSet_.addParameter<vstring>("@all_essources", empty);
    processParameterSet_.addParameter<vstring>("@all_loopers", empty);
    processParameterSet_.addParameter<vstring>("@all_modules", empty);
    processParameterSet_.addParameter<vstring>("@end_paths", empty);
    processParameterSet_.addParameter<vstring>("@paths", empty);
    processParameterSet_.addParameter<std::string>("@process_name", processName);
  }

  void
  LHEProvenanceHelper::lheAugment(lhef::LHERunInfo const* runInfo) {
    if(runInfo == nullptr) return;
    typedef std::vector<std::string> vstring;
    auto const& heprup = *runInfo->getHEPRUP();
    processParameterSet_.addParameter<int>("IDBMUP1", heprup.IDBMUP.first);
    processParameterSet_.addParameter<int>("IDBMUP2", heprup.IDBMUP.second);
    processParameterSet_.addParameter<int>("EBMUP1", heprup.EBMUP.first);
    processParameterSet_.addParameter<int>("EBMUP2", heprup.EBMUP.second);
    processParameterSet_.addParameter<int>("PDFGUP1", heprup.PDFGUP.first);
    processParameterSet_.addParameter<int>("PDFGUP2", heprup.PDFGUP.second);
    processParameterSet_.addParameter<int>("PDFSUP1", heprup.PDFSUP.first);
    processParameterSet_.addParameter<int>("PDFSUP2", heprup.PDFSUP.second);
    processParameterSet_.addParameter<int>("IDWTUP", heprup.IDWTUP);
    for(auto const& header : runInfo->getHeaders()) {
      if(!LHERunInfoProduct::isTagComparedInMerge(header.tag())) {
        continue;
      }
      processParameterSet_.addParameter<vstring>(header.tag(), header.lines());
    }
  }

  ProcessHistoryID
  LHEProvenanceHelper::lheInit(ProductRegistry& productRegistry) {
    // Now we register the process parameter set.
    processParameterSet_.registerIt();
    //std::cerr << processParameterSet_.dump() << std::endl;
    // Now we need to set all the metadata
    // Add the products to the product registry  
    productRegistry.copyProduct(eventProductBranchDescription_.me());
    productRegistry.copyProduct(runProductBranchDescription_.me());

    // Insert an entry for this process in the process history registry
    ProcessHistory ph;
    ph.emplace_back(eventProductBranchDescription_.processName(), processParameterSet_.id(), getReleaseVersion(), getPassID());
    ProcessConfiguration const& pc = ph.data().back();
    ProcessHistoryRegistry::instance()->insertMapped(ph);

    // Insert an entry for this process in the process configuration registry
    ProcessConfigurationRegistry::instance()->insertMapped(pc);

    // Save the process history ID for use every event.
    return ph.id();
  }

}
