// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServiceToken
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Sep  8 04:26:12 EDT 2005
//

// system include files

// user include files
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServicesManager.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
//ServiceToken::ServiceToken()
//{
//}

// ServiceToken::ServiceToken(const ServiceToken& rhs)
// {
//    // do actual copying here;
// }

//ServiceToken::~ServiceToken()
//{
//}

//
// assignment operators
//
// const ServiceToken& ServiceToken::operator=(const ServiceToken& rhs)
// {
//   //An exception safe implementation is
//   ServiceToken temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
edm::OwningServiceToken::OwningServiceToken(edm::ServiceToken iToken)
    : manager_(iToken.manager_ ? iToken.manager_->shared_from_this() : nullptr) {}

edm::WeakServiceToken::WeakServiceToken(edm::ServiceToken iToken)
    : manager_(iToken.manager_ ? iToken.manager_->weak_from_this()
                               : std::weak_ptr<edm::serviceregistry::ServicesManager>()) {}

void edm::OwningServiceToken::connect(edm::ActivityRegistry& iConnectTo) {
  if (manager_) {
    manager_->connect(iConnectTo);
  }
}

void edm::OwningServiceToken::copySlotsTo(edm::ActivityRegistry& iConnectTo) {
  if (manager_) {
    manager_->copySlotsTo(iConnectTo);
  }
}

//
// const member functions
//

//
// static member functions
//
