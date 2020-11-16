// -*- C++ -*-
//
// Package:     Utilities
// Class  :     cmsGetFnConnect
//
// Implementation:
//     Looks up a frontier connect string
//
// Original Author:  Dave Dykstra
//         Created:  Tue Feb 22 16:54:06 CST 2011
//

#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <cstring>
#include <memory>

int main(int argc, char* argv[]) {
  if ((argc != 2) || (strncmp(argv[1], "frontier://", 11) != 0)) {
    std::cerr << "Usage: cmsGetFnConnect frontier://shortname" << std::endl;
    return 2;
  }

  try {
    std::unique_ptr<edm::SiteLocalConfig> slcptr =
        std::make_unique<edm::service::SiteLocalConfigService>(edm::ParameterSet());
    auto slc = std::make_shared<edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig> >(std::move(slcptr));
    edm::ServiceToken slcToken = edm::ServiceRegistry::createContaining(slc);
    edm::ServiceRegistry::Operate operate(slcToken);

    edm::Service<edm::SiteLocalConfig> localconfservice;

    std::cout << localconfservice->lookupCalibConnect(argv[1]) << std::endl;
  } catch (cms::Exception const& e) {
    std::cerr << e.explainSelf() << std::endl;
    return 2;
  }
  return 0;
}
