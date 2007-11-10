#include "DataFormats/Common/interface/RefCore.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  EDProduct const*
  RefCore::getProductPtr() const {
    // The following invariant would be nice to establish in all
    // constructors, but we can not be sure that the context in which
    // EDProductGetter::instance() is called will be one where a
    // non-null pointer is returned. The biggest question in the
    // various times at which Root makes RefCore instances, and how
    // old ones might be recycled.
    //
    // If our ProductID is non-null, we must have a way to get at the
    // product (unless it has been dropped). This means either we
    // already have the pointer to the product, or we have a valid
    // EDProductGetter to use.
    //
    //     assert(!id_.isValid() || productGetter() || prodPtr_);

    if (!id_.isValid()) {
      throw Exception(errors::InvalidReference,
		      "BadRefCore")
	<< "Attempt to dereference a RefCore containing an invalid\n"
	<< "ProductID has been detected. Please modify the calling\n"
	<< "code to test validity before dereferencing.\n";
    }

    if (prodPtr_ == 0 && productGetter() == 0) {
      throw Exception(errors::InvalidReference,
		      "BadRefCore")
	<< "Attempt to dereference a RefCore containing a valid ProductID\n"
	<< "but neither a valid product pointer not EDProductGetter has\n"
	<< "been detected. The calling code must be modified to establish\n"
	<< "a functioning EDProducterGetter for the context in which this\n"
	<< "call is mode\n";
    }
    return productGetter()->getIt(id_);
  }

  bool
  RefCore::isAvailable() const {
      return prodPtr_ != 0 || (id_.isValid() && productGetter() != 0 && productGetter()->getIt(id_) != 0);
  }

//   void
//   RefCore::throwInvalidReference() const {
//     throw edm::Exception(errors::InvalidReference,"NullID")
//       << "An attempt to dereference an invalid RefCore has been detected\n"
//       << "The calling code failed to verify the validity of the RefCore.\n"
//       << "The calling code should be modified to assure validity of each\n"
//       << "RefCore object it uses, before dereferencing the RefCore\n";
//   }

  void
  checkProduct(RefCore const& productToBeInserted, RefCore & commonProduct) {
    if (productToBeInserted.isNull()) {
      throw edm::Exception(errors::InvalidReference,"Inconsistency")
	<< "RefVectorBase::push_back: Ref has invalid (zero) product ID. "
	<< "id should be (" << commonProduct.id() << ")";
    }
    if (commonProduct.isNull()) {
      commonProduct = productToBeInserted; 
    } else if (commonProduct == productToBeInserted) {
      if (commonProduct.productGetter() == 0 && productToBeInserted.productGetter() != 0) {
        commonProduct.setProductGetter(productToBeInserted.productGetter());
      }
      if (commonProduct.productPtr() == 0 && productToBeInserted.productPtr() != 0) {
        commonProduct.setProductPtr(productToBeInserted.productPtr());
      }
    } else {
      throw edm::Exception(errors::InvalidReference,"Inconsistency")
	<< "RefVectorBase::push_back: Ref is inconsistent. "
	<< "id = (" << productToBeInserted.id() << ") should be (" << commonProduct.id() << ")";
    }
  }
}
