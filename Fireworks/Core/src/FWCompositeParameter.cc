// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCompositeParameter
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:36:57 EST 2008
//

// system include files
#include <cassert>
#include <algorithm>

// user include files
#include "Fireworks/Core/interface/FWCompositeParameter.h"
#include "Fireworks/Core/interface/FWConfiguration.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWCompositeParameter::FWCompositeParameter(FWParameterizable* iParent, const std::string& iName, unsigned int iVersion)
    : FWParameterBase(iParent, iName), m_version(iVersion) {}

// FWCompositeParameter::FWCompositeParameter(const FWCompositeParameter& rhs)
// {
//    // do actual copying here;
// }

FWCompositeParameter::~FWCompositeParameter() {}

//
// assignment operators
//
// const FWCompositeParameter& FWCompositeParameter::operator=(const FWCompositeParameter& rhs)
// {
//   //An exception safe implementation is
//   FWCompositeParameter temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void FWCompositeParameter::setFrom(const FWConfiguration& iFrom) {
  //need a way to handle versioning
  const FWConfiguration* mine = iFrom.valueForKey(name());
  const FWConfiguration::KeyValues* keyVals = mine->keyValues();

  assert(nullptr != mine);
  assert(mine->version() == m_version);
  assert(nullptr != keyVals);

  for (const_iterator it = begin(), itEnd = end(); it != itEnd; ++it) {
    (*it)->setFrom(*mine);
  }
}

//
// const member functions
//
void FWCompositeParameter::addTo(FWConfiguration& oTo) const {
  FWConfiguration conf(m_version);

  for (const_iterator it = begin(), itEnd = end(); it != itEnd; ++it) {
    (*it)->addTo(conf);
  }
  //   std::for_each(begin(), end(),
  //                 std::bind(&FWParameterBase::addTo, std::placeholders::_1,conf));

  oTo.addKeyValue(name(), conf, true);
}

//
// static member functions
//
