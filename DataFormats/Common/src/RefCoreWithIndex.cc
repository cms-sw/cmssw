#include "DataFormats/Common/interface/RefCoreWithIndex.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/Common/interface/RefCore.h"
#include <cassert>
#include <iostream>
#include <ostream>

namespace edm {

  RefCoreWithIndex::RefCoreWithIndex(ProductID const& theId, void const* prodPtr, EDProductGetter const* prodGetter, bool transient, unsigned int iIndex) :
      cachePtr_(prodPtr),
      processIndex_(theId.processIndex()),
      productIndex_(theId.productIndex()),
      elementIndex_(iIndex)
      {
        if(transient) {
          setTransient();
        }
        if(prodPtr==nullptr && prodGetter!=nullptr) {
          setCacheIsProductGetter(prodGetter);
        }
      }

  RefCoreWithIndex::RefCoreWithIndex(RefCore const& iCore, unsigned int iIndex):
  cachePtr_(iCore.cachePtr_.load()),
  processIndex_(iCore.processIndex_),
  productIndex_(iCore.productIndex_),
  elementIndex_(iIndex){}
  
  RefCoreWithIndex::RefCoreWithIndex( RefCoreWithIndex const& iOther) :
    cachePtr_(iOther.cachePtr_.load()),
    processIndex_(iOther.processIndex_),
    productIndex_(iOther.productIndex_),
    elementIndex_(iOther.elementIndex_){}
  
  RefCoreWithIndex& RefCoreWithIndex::operator=( RefCoreWithIndex const& iOther) {
    cachePtr_ = iOther.cachePtr_.load();
    processIndex_ = iOther.processIndex_;
    productIndex_ = iOther.productIndex_;
    elementIndex_ = iOther.elementIndex_;
    return *this;
  }

}

