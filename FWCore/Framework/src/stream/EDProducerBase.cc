// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::stream::EDProducerBase
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 23:49:57 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/EDProducerBase.h"
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
EDProducerBase::EDProducerBase() : moduleDescriptionPtr_(nullptr) {}

// EDProducerBase::EDProducerBase(const EDProducerBase& rhs)
// {
//    // do actual copying here;
// }

EDProducerBase::~EDProducerBase() {}

//
// assignment operators
//
// const EDProducerBase& EDProducerBase::operator=(const EDProducerBase& rhs)
// {
//   //An exception safe implementation is
//   EDProducerBase temp(rhs);
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
void EDProducerBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

size_t EDProducerBase::transformIndex_(edm::BranchDescription const& iBranch) const { return -1; }
edm::ProductResolverIndex EDProducerBase::transformPrefetch_(std::size_t iIndex) const { return 0; }
void EDProducerBase::transformAsync_(WaitingTaskHolder iTask,
                                     std::size_t iIndex,
                                     edm::EventForTransformer& iEvent,
                                     ServiceWeakToken const& iToken) const {}

void EDProducerBase::prevalidate(ConfigurationDescriptions& iConfig) { edmodule_mightGet_config(iConfig); }

static const std::string kBaseType("EDProducer");

const std::string& EDProducerBase::baseType() { return kBaseType; }
