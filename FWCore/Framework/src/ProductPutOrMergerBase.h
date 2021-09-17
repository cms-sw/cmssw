//
//  ProductPutOrMergerBase.h
//  CMSSW
//
//  Created by Chris Jones on 3/18/21.
//

#ifndef FWCore_Framework_ProductPutOrMergerBase_h
#define FWCore_Framework_ProductPutOrMergerBase_h

#include <memory>

namespace edm {
  class WrapperBase;
  class MergeableRunProductMetadata;

  class ProductPutOrMergerBase {
  public:
    ProductPutOrMergerBase() = default;
    virtual ~ProductPutOrMergerBase() = default;

    virtual void putOrMergeProduct(std::unique_ptr<WrapperBase> edp) const = 0;
  };
}  // namespace edm

#endif /* ProductPutOrMergerBase_h */
