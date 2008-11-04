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
// $Id: PtrVectorBase.cc,v 1.4 2008/02/15 20:06:21 wmtan Exp $
//

// system include files

// user include files
#include "DataFormats/Common/interface/PtrVectorBase.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/traits.h"
#include <ostream>

//
// constants, enums and typedefs
//
namespace edm {
//
// static data member definitions
//

//
// constructor and destructor
//
  PtrVectorBase::PtrVectorBase()
  {
  }

  PtrVectorBase::~PtrVectorBase()
  {
  }

//
// assignment operators
//

//
// member functions
//

  /// swap
  void
  PtrVectorBase::swap(PtrVectorBase& other){
    core_.swap(other.core_);
    indicies_.swap(other.indicies_);
    cachedItems_.swap(other.cachedItems_);
  }

  void 
  PtrVectorBase::push_back_base(RefCore const& core, key_type iKey, void const* iData) {
    core_.pushBackItem(core, false);
    //Did we already push a 'non-cached' Ptr into the container or is this a 'non-cached' Ptr?
    if(indicies_.size() == cachedItems_.size()) {
      if(iData) {
        cachedItems_.push_back(iData);
      } else if(key_traits<key_type>::value == iKey) {
        cachedItems_.push_back(0);
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
  PtrVectorBase::getProduct_() const {
    if(hasCache()) {
      return;
    }
    if(indicies_.size() == 0) {
      return;
    }
    if(0 == productGetter()) {
      throw edm::Exception(edm::errors::LogicError) << "Tried to get data for a PtrVector which has no EDProductGetter\n";
    }
    EDProduct const* product = productGetter()->getIt(id());

    if(0==product) {
      throw edm::Exception(edm::errors::InvalidReference) << "Asked for data from a PtrVector which refers to a non-existent product which id " << id() << "\n";
    }
    product->fillPtrVector(typeInfo(),
                              indicies_,
                              cachedItems_);
  }
  
  bool 
  PtrVectorBase::operator==(PtrVectorBase const& iRHS) const {
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
