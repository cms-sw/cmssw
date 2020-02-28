// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemSingleAccessor
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 18 11:36:44 EDT 2008
//

// system include files
#include <cassert>
#include "FWCore/Reflection/interface/ObjectWithDict.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"

// user include files
#include "Fireworks/Core/src/FWItemSingleAccessor.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWItemSingleAccessor::FWItemSingleAccessor(const TClass* iClass) : m_type(iClass), m_data(nullptr) {}

// FWItemSingleAccessor::FWItemSingleAccessor(const FWItemSingleAccessor& rhs)
// {
//    // do actual copying here;
// }

FWItemSingleAccessor::~FWItemSingleAccessor() {}

//
// assignment operators
//
// const FWItemSingleAccessor& FWItemSingleAccessor::operator=(const FWItemSingleAccessor& rhs)
// {
//   //An exception safe implementation is
//   FWItemSingleAccessor temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void FWItemSingleAccessor::setData(const edm::ObjectWithDict& product) {
  if (product.address() == nullptr) {
    reset();
    return;
  }

  m_data = product.address();
  assert(nullptr != m_data);
}

void FWItemSingleAccessor::reset() { m_data = nullptr; }

//
// const member functions
//
const void* FWItemSingleAccessor::modelData(int iIndex) const {
  if (0 == iIndex) {
    return m_data;
  }
  return nullptr;
}

const void* FWItemSingleAccessor::data() const { return m_data; }

unsigned int FWItemSingleAccessor::size() const { return nullptr == m_data ? 0 : 1; }

const TClass* FWItemSingleAccessor::modelType() const { return m_type; }

const TClass* FWItemSingleAccessor::type() const { return m_type; }

bool FWItemSingleAccessor::isCollection() const { return false; }

//
// static member functions
//
