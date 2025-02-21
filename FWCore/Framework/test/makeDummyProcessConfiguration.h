#ifndef FWCore_Framework_test_makeDummyProcessConfiguration_h
#define FWCore_Framework_test_makeDummyProcessConfiguration_h

#include "DataFormats/Provenance/interface/HardwareResourcesDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include <memory>

namespace edmtest {
  inline edm::ProcessConfiguration makeDummyProcessConfiguration(
      std::string const& processName, edm::ParameterSetID const& pSetID = edm::ParameterSetID()) {
    return edm::ProcessConfiguration(
        processName, pSetID, edm::getReleaseVersion(), edm::HardwareResourcesDescription());
  }

  inline std::shared_ptr<edm::ProcessConfiguration> makeSharedDummyProcessConfiguration(
      std::string const& processName, edm::ParameterSetID const& pSetID = edm::ParameterSetID()) {
    return std::make_shared<edm::ProcessConfiguration>(
        processName, pSetID, edm::getReleaseVersion(), edm::HardwareResourcesDescription());
  }
}  // namespace edmtest

#endif
