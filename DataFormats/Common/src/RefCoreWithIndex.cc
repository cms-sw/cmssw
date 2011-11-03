#include "DataFormats/Common/interface/RefCoreWithIndex.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/Common/interface/RefCore.h"
#include <cassert>
#include <iostream>
#include <ostream>

namespace edm {

  RefCoreWithIndex::RefCoreWithIndex(ProductID const& theId, void const* prodPtr, EDProductGetter const* prodGetter, bool transient, unsigned int iIndex) :
      cachePtr_(prodPtr?prodPtr:prodGetter),
      processIndex_(theId.processIndex()),
      productIndex_(theId.productIndex()),
      elementIndex_(iIndex)
      {
        if(transient) {
          setTransient();
        }
        if(prodPtr!=0 || prodGetter==0) {
          setCacheIsProductPtr();
        }
      }

  RefCoreWithIndex::RefCoreWithIndex(RefCore const& iCore, unsigned int iIndex):
  cachePtr_(iCore.cachePtr_),
  processIndex_(iCore.processIndex_),
  productIndex_(iCore.productIndex_),
  elementIndex_(iIndex){}
  
}
