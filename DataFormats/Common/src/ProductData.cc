/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/ProductData.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"

#include <algorithm>

namespace edm {
  ProductData::ProductData() : wrapper_(), prov_() {}

  ProductData::ProductData(std::shared_ptr<ProductDescription const> bd) : wrapper_(), prov_(bd, ProductID()) {}

  // For use by FWLite
  ProductData::ProductData(WrapperBase* product, Provenance const& prov)
      : wrapper_(product, do_nothing_deleter()), prov_(prov) {}

  void ProductData::resetProductDescription(std::shared_ptr<ProductDescription const> bd) {
    prov_.setProductDescription(bd);
  }

  void ProductData::setWrapper(std::unique_ptr<WrapperBase> iValue) { wrapper_ = std::move(iValue); }

  //Not const thread-safe update
  void ProductData::unsafe_setWrapper(std::unique_ptr<WrapperBase> iValue) const { wrapper_ = std::move(iValue); }

  void ProductData::unsafe_setWrapper(std::shared_ptr<WrapperBase const> iValue) const { wrapper_ = std::move(iValue); }
}  // namespace edm
