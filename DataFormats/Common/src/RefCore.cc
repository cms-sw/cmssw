#include "DataFormats/Common/interface/RefCore.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  void
  RefCore::badID() const {
    throw edm::Exception(errors::InvalidReference,"BadID")
      << "RefCore::RefCore: Ref initialized with zero id.";
  }

  void
  RefCore::nullID() const {
    throw edm::Exception(errors::InvalidReference,"NullID")
      << "RefCore::getProduct: Attempt to use a null Ref.";
  }

  void
  wrongRefType(std::string const& found, std::string const& requested) {
    throw edm::Exception(errors::InvalidReference,"WrongType")
      << "getProduct_<T>: Collection is of wrong type:\n"
      << "found type=" << found << "\n"
      << "requested type=" << requested << "\n";
  }

  void
  checkProduct(RefCore const& prod, RefCore & product) {
    if (product.id() == ProductID()) {
      product = prod; 
    } else if (product == prod) {
      if (product.productGetter() == 0 && prod.productGetter() != 0) {
        product.setProductGetter(prod.productGetter());
      }
      if (product.productPtr() == 0 && prod.productPtr() != 0) {
        product.setProductPtr(prod.productPtr());
      }
    } else {
      throw edm::Exception(errors::InvalidReference,"Inconsistency")
	<< "RefVectorBase::push_back: Ref is inconsistent. "
	<< "id = (" << prod.id() << ") should be (" << product.id() << ")";
    }
  }
}
