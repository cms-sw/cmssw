// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::stream::EDFilterBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 23:49:57 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/EDFilterBase.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

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
EDFilterBase::EDFilterBase()
{
}

// EDFilterBase::EDFilterBase(const EDFilterBase& rhs)
// {
//    // do actual copying here;
// }

EDFilterBase::~EDFilterBase()
{
}

//
// assignment operators
//
// const EDFilterBase& EDFilterBase::operator=(const EDFilterBase& rhs)
// {
//   //An exception safe implementation is
//   EDFilterBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
void
EDFilterBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void
EDFilterBase::prevalidate(ConfigurationDescriptions& iConfig) {
  edmodule_mightGet_config(iConfig);
}

static const std::string kBaseType("EDFilter");

const std::string&
EDFilterBase::baseType() {
  return kBaseType;
}
