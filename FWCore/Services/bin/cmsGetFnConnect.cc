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
#include "FWCore/Services/interface/setupSiteLocalConfig.h"
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
    auto operate = edm::setupSiteLocalConfig();

    edm::Service<edm::SiteLocalConfig> localconfservice;

    std::cout << localconfservice->lookupCalibConnect(argv[1]) << std::endl;
  } catch (cms::Exception const& e) {
    std::cerr << e.explainSelf() << std::endl;
    return 2;
  }
  return 0;
}
