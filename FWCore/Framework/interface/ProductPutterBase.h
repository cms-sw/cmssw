//
//  ProductPutterBase.h
//  CMSSW
//
//  Created by Chris Jones on 3/18/21.
//

#ifndef FWCore_Framework_ProductPutterBase_h
#define FWCore_Framework_ProductPutterBase_h

#include <memory>

namespace edm {
  class WrapperBase;

  class ProductPutterBase {
  public:
    ProductPutterBase() = default;
    virtual ~ProductPutterBase() = default;

    // Puts the product into the ProductResolver.
    virtual void putProduct(std::unique_ptr<WrapperBase> edp) const = 0;
  };
}  // namespace edm

#endif /* ProductPutterBase_h */
