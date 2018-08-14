#include "FWCore/Framework/interface/getProducerParameterSet.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>

namespace edm {

  ParameterSet const*
  getProducerParameterSet(Provenance const& provenance) {

    const std::shared_ptr<BranchDescription const>& branchDescription =
      provenance.constBranchDescriptionPtr();

    if (branchDescription) {
      std::string const& process = branchDescription->processName();
      std::string const& label = branchDescription->moduleLabel();
      ProcessHistory const* processHistory = provenance.processHistoryPtr();

      if (processHistory) {
        for (ProcessConfiguration const& pc : *processHistory) {
          if (pc.processName() == process) {
            ParameterSetID const& psetID = pc.parameterSetID();
            pset::Registry const* psetRegistry = pset::Registry::instance();
            ParameterSet const* processPset = psetRegistry->getMapped(psetID);
            if (processPset) {
              return &processPset->getParameterSet(label);
            }
          }
        }
      }
    }
    // This should never happen
    throw cms::Exception("LogicError")
      << "getProducerParameterSet failed";
    return nullptr;
  }
}
