// -*- C++ -*-
//
// Package:     FWCore/Services
// Class  :     ProductRegistryDumper
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Thu, 23 Mar 2017 18:32:17 GMT
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// system include files

// user include files

namespace edm {
  namespace service {
    class ProductRegistryDumper {
    public:
      ProductRegistryDumper(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iAR);
    };

  }  // namespace service
}  // namespace edm

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
using namespace edm::service;
ProductRegistryDumper::ProductRegistryDumper(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iAR) {
  iAR.watchPostBeginJob([]() {
    Service<ConstProductRegistry> regService;
    for (auto const& branch : regService->allBranchDescriptions()) {
      if (branch) {
        edm::LogSystem("ProductRegistry") << *branch;
      }
    }
  });
}

DEFINE_FWK_SERVICE(ProductRegistryDumper);
