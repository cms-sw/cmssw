#include "DataFormats/Common/interface/RefCore.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <cassert>
#include <ostream>

namespace edm {

  RefCore::RefCore(ProductID const& theId, void const* prodPtr, EDProductGetter const* prodGetter, bool transient) :
      id_(theId), 
      transients_(prodPtr, prodGetter, transient) {}

  EDProduct const*
  RefCore::getProductPtr(char const* type) const {
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

    assert (!isTransient());
    if (!id_.isValid()) {
      throw Exception(errors::InvalidReference,
		      "BadRefCore")
	<< "Attempt to dereference a RefCore containing an invalid\n"
	<< "ProductID has been detected. Please modify the calling\n"
	<< "code to test validity before dereferencing.\n";
    }

    if (productPtr() == 0 && productGetter() == 0) {
      throw Exception(errors::InvalidReference,
		      "BadRefCore")
	<< "Attempt to dereference a RefCore containing a valid ProductID\n"
	<< "but neither a valid product pointer not EDProductGetter has\n"
	<< "been detected. The calling code must be modified to establish\n"
	<< "a functioning EDProducterGetter for the context in which this\n"
	<< "call is mode\n";
    }
    EDProduct const* product = productGetter()->getIt(id_);
    if (product == 0) {
      throw edm::Exception(errors::ProductNotFound)
	<< "RefCore: A request to resolve a reference to a product of type: "
	<< type
        << "with ProductID "<< id_
	<< "\ncan not be satisfied because the product cannot be found."
	<< "\nProbably the branch containing the product is not stored in the input file.\n";
    }
    return product;
  }

  void
  RefCore::wrongTypeException(char const* expectedType, char const* actualType) const {
    throw edm::Exception(errors::InvalidReference,"WrongType")
	<< "RefCore: A request to convert a contained product of type: "
	<<  actualType << "\n"
	<< " to type " << expectedType
	<< "\nfor ProductID "<< id_
	<< " can not be satisfied\n";
  }

  void
  RefCore::nullPointerForTransientException(char const* type) const {
    throw edm::Exception(errors::ProductNotFound)
	<< "RefCore: A request to resolve a transient reference to a product of type: "
	<< type
	<< "\ncan not be satisfied because the pointer to the product is null.\n";
  }

  bool
  RefCore::isAvailable() const {
      return productPtr() != 0 || (id_.isValid() && productGetter() != 0 && productGetter()->getIt(id_) != 0);
  }

  void
  RefCore::setProductGetter(EDProductGetter const* prodGetter) const {
    transients_.setProductGetter(prodGetter);
  }

  void
  RefCore::RefCoreTransients::setProductGetter(EDProductGetter const* prodGetter) const {
    if (!transient_) {
      prodGetter_ = prodGetter;
    }
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
  RefCore::pushBackItem(RefCore const& productToBeInserted, bool checkPointer) {
    if (productToBeInserted.isNull() && !productToBeInserted.isTransient()) {
      throw edm::Exception(errors::InvalidReference,"Inconsistency")
	<< "RefCore::pushBackItem: Ref or Ptr has invalid (zero) product ID, so it cannot be added to RefVector (PtrVector). "
	<< "id should be (" << id() << ")\n";
    }
    if (isNonnull()) {
      if (isTransient() != productToBeInserted.isTransient()) {
        if (productToBeInserted.isTransient()) {
	  throw edm::Exception(errors::InvalidReference,"Inconsistency")
	    << "RefCore::pushBackItem: Transient Ref or Ptr cannot be added to persistable RefVector (PtrVector). "
	    << "id should be (" << id() << ")\n";
        } else {
	  throw edm::Exception(errors::InvalidReference,"Inconsistency")
	    << "RefCore::pushBackItem: Persistable Ref or Ptr cannot be added to transient RefVector (PtrVector). "
	    << "id is (" << productToBeInserted.id() << ")\n";
        }
      }
      if (!productToBeInserted.isTransient() && id() != productToBeInserted.id()) {
        throw edm::Exception(errors::InvalidReference,"Inconsistency")
	  << "RefCore::pushBackItem: Ref or Ptr is inconsistent with RefVector (PtrVector)"
	  << "id = (" << productToBeInserted.id() << ") should be (" << id() << ")\n";
      }
      if (productToBeInserted.isTransient() && checkPointer && productToBeInserted.isNonnull() && productToBeInserted != *this) {
        throw edm::Exception(errors::InvalidReference,"Inconsistency")
	   << "RefCore::pushBackItem: Ref points into different collection than the RefVector.\n";
      }
    } else {
      if (productToBeInserted.isTransient()) {
        setTransient();
      }
      if (productToBeInserted.isNonnull()) {
        setId(productToBeInserted.id());
      }
    }
    if (productGetter() == 0 && productToBeInserted.productGetter() != 0) {
      setProductGetter(productToBeInserted.productGetter());
    }
    if (productPtr() == 0 && productToBeInserted.productPtr() != 0) {
      setProductPtr(productToBeInserted.productPtr());
    }
  }
}
