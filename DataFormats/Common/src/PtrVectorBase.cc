// -*- C++ -*-
//
// Package:     Common
// Class  :     PtrVectorBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Oct 24 15:49:27 EDT 2007
// $Id$
//

// system include files

// user include files
#include "DataFormats/Common/interface/PtrVectorBase.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/traits.h"

//
// constants, enums and typedefs
//
namespace edm {
//
// static data member definitions
//

//
// constructors and destructor
//
  PtrVectorBase::PtrVectorBase()
  {
  }

// PtrVectorBase::PtrVectorBase(const PtrVectorBase& rhs)
// {
//    // do actual copying here;
// }

  PtrVectorBase::~PtrVectorBase()
  {
  }

//
// assignment operators
//
// const PtrVectorBase& PtrVectorBase::operator=(const PtrVectorBase& rhs)
// {
//   //An exception safe implementation is
//   PtrVectorBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
  void 
  PtrVectorBase::push_back_base(key_type iKey, 
                                const void* iData,
                                const ProductID& iID,
                                const EDProductGetter* iGetter) {
    //Test that the item belongs in our collection
    if(id() != ProductID()) {
      if(iID !=id()) {
        throw cms::Exception("ProductMissmatch")<<"attempted to put an edm::Ptr from a collection with id "
        <<iID<< " into a PtrVector associated with a collection with id "<<id();
      }
    } else {
      core_.setId(iID);
    }
    if( !productGetter() ) {
      core_.setProductGetter(iGetter);
    }
    //Did we already push a 'non-cached' Ptr into the container or is this a 'non-cached' Ptr?
    if(indicies_.size() == cachedItems_.size()
       && key_traits<key_type>::value != iKey)
    {
      if(iData) {
        cachedItems_.push_back(iData);
      } else {
        cachedItems_.clear();
      }
    }
    indicies_.push_back(iKey);
  }

//
// const member functions
//
  void 
  PtrVectorBase::getProduct_() const
  {
    if(cachedItems_.size()) {
      return;
    }
    if(indicies_.size() == 0) {
      return;
    }
    if(0 == productGetter()) {
      throw edm::Exception(edm::errors::LogicError) <<"Tried to get data for a PtrVector which has no EDProductGetter";  
    }
    const EDProduct* product = productGetter()->getIt(id());

    if(0==product) {
      throw edm::Exception(edm::errors::InvalidReference) <<"Asked for data from a PtrVector which refers to a non-existent product which id "<<id();
    }
    product->fillPtrVector(typeInfo(),
                              indicies_,
                              cachedItems_);
  }
  
  bool 
  PtrVectorBase::operator==(const PtrVectorBase& iRHS) const {
    if (core_ != iRHS.core_) {
      return false;
    }
    if (indicies_.size() != iRHS.indicies_.size()){
      return false;
    }
    return std::equal(indicies_.begin(),
                      indicies_.end(),
                      iRHS.indicies_.begin());
    return true;
  }

//
// static member functions
//
}
