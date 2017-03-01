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
  PtrVectorBase::PtrVectorBase(): cachedItems_(nullptr) {
  }

  PtrVectorBase::~PtrVectorBase() {
    delete cachedItems_.load();
  }

  PtrVectorBase::PtrVectorBase( const PtrVectorBase& iOther):
  core_(iOther.core_),
  indicies_(iOther.indicies_),
  cachedItems_(nullptr)
  {
    auto cache = iOther.cachedItems_.load();
    if(cache) {
      cachedItems_.store( new std::vector<void const*>(*cache));
    }
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
    other.cachedItems_.store(cachedItems_.exchange(other.cachedItems_.load()));
  }

  void
  PtrVectorBase::push_back_base(RefCore const& core, key_type iKey, void const* iData) {
    core_.pushBackItem(core, false);
    //Did we already push a 'non-cached' Ptr into the container or is this a 'non-cached' Ptr?
    if(not cachedItems_ and indicies_.empty()) {
      cachedItems_.store( new std::vector<void const*>());
      (*cachedItems_).reserve(indicies_.capacity());
    }
    auto tmpCachedItems = cachedItems_.load();
    if(tmpCachedItems and (indicies_.size() == (*tmpCachedItems).size())) {
      if(iData) {
        tmpCachedItems->push_back(iData);
      } else if(key_traits<key_type>::value == iKey) {
        tmpCachedItems->push_back(nullptr);
      } else {
        delete tmpCachedItems;
        cachedItems_.store(nullptr);
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
    auto tmpCachedItems = cachedItems_.load();
    for(auto ptr : *tmpCachedItems) {
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
    //NOTE: Another thread could be getting the data
    auto tmpProductGetter = productGetter();
    if(nullptr == tmpProductGetter) {
      throw Exception(errors::LogicError) << "Tried to get data for a PtrVector which has no EDProductGetter\n";
    }
    WrapperBase const* product = tmpProductGetter->getIt(id());

    auto tmpCachedItems = std::make_unique<std::vector<void const*>>();

    if(product != nullptr) {
      product->fillPtrVector(typeInfo(), indicies_, *tmpCachedItems);
      
      std::vector<void const*>* expected = nullptr;
      if(cachedItems_.compare_exchange_strong(expected, tmpCachedItems.get())) {
        //we were the first thread to change the value
        tmpCachedItems.release();
      }

      return;
    }
    
    tmpCachedItems->resize(indicies_.size(), nullptr);

    std::vector<unsigned int> thinnedKeys;
    thinnedKeys.assign(indicies_.begin(), indicies_.end());
    std::vector<WrapperBase const*> wrappers(indicies_.size(), nullptr);
    tmpProductGetter->getThinnedProducts(id(), wrappers, thinnedKeys);
    unsigned int nWrappers = wrappers.size();
    assert(wrappers.size() == indicies_.size());
    assert(wrappers.size() == tmpCachedItems->size());
    for(unsigned k = 0; k < nWrappers; ++k) {
      if (wrappers[k] != nullptr) {
        wrappers[k]->setPtr(typeInfo(), thinnedKeys[k], (*tmpCachedItems)[k]);
      }
    }
    {
      std::vector<void const*>* expected = nullptr;
      if(cachedItems_.compare_exchange_strong(expected, tmpCachedItems.get())) {
        //we were the first thread to change the value
        tmpCachedItems.release();
      }
    }
  }

  bool
  PtrVectorBase::checkCachedItems() const {
    auto tmp = cachedItems_.load();
    if(not tmp) { return false;}
    for(auto item : *tmp) {
      if(item == nullptr) {
        throw Exception(errors::InvalidReference) << "Asked for data from a PtrVector which refers to a non-existent product with ProductID "
                                                  << id() << "\n";
      }
    }
    return true;
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

  static const std::vector<void const*> s_emptyCache{};
  
  const std::vector<void const*>& PtrVectorBase::emptyCache() {
    return s_emptyCache;
  }
  
}
