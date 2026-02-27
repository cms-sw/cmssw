// -*- C++ -*-
//
// Package:     FWCore/Reflection
// Class  :     DataProductReflectionInfoRegistry
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 27 Jul 2022 21:12:54 GMT
//

// system include files

// user include files
#include "FWCore/Reflection/interface/DataProductReflectionInfoRegistry.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
edm::DataProductReflectionInfoRegistry::DataProductReflectionInfoRegistry() {}

edm::DataProductReflectionInfoRegistry::~DataProductReflectionInfoRegistry() {}

//
// member functions
//
void edm::DataProductReflectionInfoRegistry::registerDataProduct(std::type_index iIndex,
                                                                 DataProductReflectionInfo iInfo) {
  m_registry.emplace(iIndex, iInfo);
}

//
// const member functions
//
edm::DataProductReflectionInfo const* edm::DataProductReflectionInfoRegistry::findType(std::type_index iIndex) const {
  auto itFound = m_registry.find(iIndex);
  if (itFound == m_registry.end()) {
    return nullptr;
  }
  return &(itFound->second);
}

//
// static member functions
//
edm::DataProductReflectionInfoRegistry& edm::DataProductReflectionInfoRegistry::instance() {
  static DataProductReflectionInfoRegistry s_registry;
  return s_registry;
}
