// -*- C++ -*-
//
// Package:     EDProduct
// Class  :     EDProductGetter
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov  1 15:06:41 EST 2005
// $Id: EDProductGetter.cc,v 1.3 2007/01/23 00:25:53 wmtan Exp $
//

// system include files
#include "boost/thread/tss.hpp"

// user include files
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  //
  // constants, enums and typedefs
  //
  
  //
  // static data member definitions
  //
  
  //
  // constructors and destructor
  //
  EDProductGetter::EDProductGetter()
  {
  }
  
  // EDProductGetter::EDProductGetter(const EDProductGetter& rhs)
  // {
  //    // do actual copying here;
  // }
  
  EDProductGetter::~EDProductGetter()
  {
  }
  
  //
  // assignment operators
  //
  // const EDProductGetter& EDProductGetter::operator=(const EDProductGetter& rhs)
  // {
  //   //An exception safe implementation is
  //   EDProductGetter temp(rhs);
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
  
  namespace {
     struct Holder {
        Holder(): held_(0) {}
        edm::EDProductGetter const* held_;
     };
  }
  EDProductGetter const* 
  EDProductGetter::set(EDProductGetter const* iGetter)
  {
     //NOTE: I use a Holder so that when s_registry goes away it will not delete the last EDProductGetter it saw
     static boost::thread_specific_ptr<Holder> s_registry;
     if(0 == s_registry.get()){
        s_registry.reset(new Holder);
     }
     EDProductGetter const* previous = s_registry->held_;
     s_registry->held_= iGetter;
     return previous;
  }
  
  EDProductGetter const*
  EDProductGetter::instance()
  {
     EDProductGetter const* returnValue = EDProductGetter::set(0);
     EDProductGetter::set(returnValue);
     return returnValue;
  }
  
  EDProductGetter const*
  mustBeNonZero(EDProductGetter const* prodGetter, std::string refType, ProductID const& productID) {
    if (prodGetter != 0) return prodGetter;
        throw Exception(errors::InvalidReference, refType)
  	<< "Attempt to construct a " << refType << " with ProductID " << productID << "\n"
  	<< "but with a null pointer to a product getter.\n"
  	<< "The product getter pointer passed to the constructor must refer\n"
  	<< "to a real getter, such as an EventPrincipal.\n";
  }
}
