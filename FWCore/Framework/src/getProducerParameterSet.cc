#include "FWCore/Framework/interface/getProducerParameterSet.h"

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>

namespace edm {

  ParameterSet const* getProducerParameterSet(Provenance const& provenance, ProcessHistory const& processHistory) {
    const std::shared_ptr<ProductDescription const>& productDescription = provenance.constProductDescriptionPtr();

    if (productDescription) {
      std::string const& process = productDescription->processName();
      std::string const& label = productDescription->moduleLabel();

      for (ProcessConfiguration const& pc : processHistory) {
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
    // This should never happen
    throw cms::Exception("LogicError") << "getProducerParameterSet failed";
    return nullptr;
  }
}  // namespace edm
