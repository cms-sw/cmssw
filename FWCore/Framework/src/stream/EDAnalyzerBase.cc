// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::stream::EDAnalyzerBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 23:49:57 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/EDAnalyzerBase.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"

using namespace edm::stream;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EDAnalyzerBase::EDAnalyzerBase(): moduleDescriptionPtr_(nullptr)
{
}

// EDAnalyzerBase::EDAnalyzerBase(const EDAnalyzerBase& rhs)
// {
//    // do actual copying here;
// }

EDAnalyzerBase::~EDAnalyzerBase()
{
}

//
// assignment operators
//
// const EDAnalyzerBase& EDAnalyzerBase::operator=(const EDAnalyzerBase& rhs)
// {
//   //An exception safe implementation is
//   EDAnalyzerBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
EDAnalyzerBase::callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func) {
  callWhenNewProductsRegistered_ = func;
}

void
EDAnalyzerBase::registerProductsAndCallbacks(EDAnalyzerBase const*, ProductRegistry* reg) {
  
  if (callWhenNewProductsRegistered_) {
    
    reg->callForEachBranch(callWhenNewProductsRegistered_);
    
    Service<ConstProductRegistry> regService;
    regService->watchProductAdditions(callWhenNewProductsRegistered_);
  }
}

//
// const member functions
//

//
// static member functions
//
void
EDAnalyzerBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void
EDAnalyzerBase::prevalidate(ConfigurationDescriptions& iConfig) {
  edmodule_mightGet_config(iConfig);
}

static const std::string kBaseType("EDAnalyzer");

const std::string&
EDAnalyzerBase::baseType() {
  return kBaseType;
}
