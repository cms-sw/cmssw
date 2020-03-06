#include "FWCore/Framework/interface/getProducerParameterSet.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {

  ParameterSet const* getProducerParameterSet(Provenance const& provenance, ProcessHistory const& processHistory) {
    const std::shared_ptr<BranchDescription const>& branchDescription = provenance.constBranchDescriptionPtr();

    if (branchDescription) {
      std::string const& process = branchDescription->processName();
      std::string const& label = branchDescription->moduleLabel();
      return getProducerParameterSet( processHistory, process, label );
    }
    // This should never happen
    throw cms::Exception("LogicError") << "getProducerParameterSet failed";
    return nullptr;
  }

  ParameterSet const* getProducerParameterSet(ProcessHistory const& processHistory, std::string const& processName, std::string const& moduleLabel) {
    ParameterSet   const* iConfig      = nullptr;
    pset::Registry const* psetRegistry = pset::Registry::instance();
    for ( ProcessConfiguration const& pc : processHistory ) {
      if ( processName != "" && processName != pc.processName() )
        continue;
      ParameterSet const* processPset = psetRegistry->getMapped( pc.parameterSetID() );
      if ( processPset && processPset->exists( moduleLabel ) )
        iConfig = &processPset->getParameterSet( moduleLabel );
    }
    if ( !iConfig )
      throw cms::Exception("Configuration") << "Failed to find module " << moduleLabel << " in process " << processName << ".";
    return iConfig;
  }

}  // namespace edm
