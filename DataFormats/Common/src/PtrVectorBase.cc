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
//

// user include files
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/PtrVectorBase.h"
#include "DataFormats/Common/interface/traits.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"

// system include files
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
  PtrVectorBase::PtrVectorBase() {
  }

  PtrVectorBase::~PtrVectorBase() {
  }

//
// assignment operators
//

//
// member functions
//

  /// swap
  void
  PtrVectorBase::swap(PtrVectorBase& other) {
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

  bool
  PtrVectorBase::isAvailable() const {
    if(indicies_.empty()) {
      return core_.isAvailable();
    }
    if(!hasCache()) {
      if(!id().isValid() || productGetter() == nullptr) {
        return false;
      }
      getProduct_();
    }
    for(auto ptr : cachedItems_) {
      if(ptr == nullptr) {
        return false;
      }
    }
    return true;
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
      throw Exception(errors::LogicError) << "Tried to get data for a PtrVector which has no EDProductGetter\n";
    }
    WrapperBase const* product = productGetter()->getIt(id());

    if(product != nullptr) {
      product->fillPtrVector(typeInfo(), indicies_, cachedItems_);
      return;
    }

    cachedItems_.resize(indicies_.size(), nullptr);

    std::vector<unsigned int> thinnedKeys;
    thinnedKeys.assign(indicies_.begin(), indicies_.end());
    std::vector<WrapperBase const*> wrappers(indicies_.size(), nullptr);
    productGetter()->getThinnedProducts(id(), wrappers, thinnedKeys);
    unsigned int nWrappers = wrappers.size();
    assert(wrappers.size() == indicies_.size());
    assert(wrappers.size() == cachedItems_.size());
    for(unsigned k = 0; k < nWrappers; ++k) {
      if (wrappers[k] != nullptr) {
        wrappers[k]->setPtr(typeInfo(), thinnedKeys[k], cachedItems_[k]);
      }
    }
  }

  void
  PtrVectorBase::checkCachedItems() const {
    for(auto item : cachedItems_) {
      if(item == nullptr) {
        throw Exception(errors::InvalidReference) << "Asked for data from a PtrVector which refers to a non-existent product with ProductID "
                                                  << id() << "\n";
      }
    }
  }

  bool
  PtrVectorBase::operator==(PtrVectorBase const& iRHS) const {
    if(core_ != iRHS.core_) {
      return false;
    }
    if(indicies_.size() != iRHS.indicies_.size()){
      return false;
    }
    return std::equal(indicies_.begin(),
                      indicies_.end(),
                      iRHS.indicies_.begin());
  }

//
// static member functions
//
}
