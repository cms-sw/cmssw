#include "DataFormats/Common/interface/RefCore.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include <cassert>
#include <iostream>
#include <ostream>

static
void throwInvalidRefFromNullOrInvalidRef(const edm::TypeID& id) {
  throw edm::Exception(edm::errors::InvalidReference,
                  "BadRefCore")
  << "RefCore: Request to resolve a null or invalid reference to a product of type '"
  << id
	<< "' has been detected.\n"
  << "Please modify the calling code to test validity before dereferencing.\n";  
}

static
void throwInvalidRefFromNoCache(const edm::TypeID& id, edm::ProductID const& prodID) {
  throw edm::Exception(edm::errors::InvalidReference,
                  "BadRefCore")
  << "RefCore: A request to resolve a reference to a product of type '"
  << id
  << "' with ProductID '" << prodID
	<< "' cannot be satisfied.\n"
	<< "The reference has neither a valid product pointer nor an EDProductGetter.\n"
	<< "The calling code must be modified to establish a functioning EDProducterGetter\n"
  << "for the context in which this call is mode\n";
  
}

namespace edm {

  RefCore::RefCore(ProductID const& theId, void const* prodPtr, EDProductGetter const* prodGetter, bool transient) :
      cachePtr_(prodPtr?prodPtr:prodGetter),
      processIndex_(theId.processIndex()),
      productIndex_(theId.productIndex()),
      transient_(transient,prodPtr!=0 || prodGetter==0)
      {}

  WrapperHolder
  RefCore::getProductPtr(std::type_info const& type) const {
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

    ProductID tId = id();
    assert (!isTransient());
    if (!tId.isValid()) {
      throwInvalidRefFromNullOrInvalidRef(TypeID(type));
    }

    //if (productPtr() == 0 && productGetter() == 0) {
    if (cachePtr_ == 0) {
      throwInvalidRefFromNoCache(TypeID(type),tId);
    }
    WrapperHolder product = productGetter()->getIt(tId);
    if (!product.isValid()) {
      productNotFoundException(type);
    }
    if(!(type == product.dynamicTypeInfo())) {
      wrongTypeException(type, product.dynamicTypeInfo());
    }
    return product;
  }

  void
  RefCore::productNotFoundException(std::type_info const& type) const {
    throw edm::Exception(errors::ProductNotFound)
      << "RefCore: A request to resolve a reference to a product of type '"
      << TypeID(type)
      << "' with ProductID '" << id() << "'"
      << "\ncan not be satisfied because the product cannot be found."
      << "\nProbably the branch containing the product is not stored in the input file.\n";
  }

  void
  RefCore::wrongTypeException(std::type_info const& expectedType, std::type_info const& actualType) const {
    throw edm::Exception(errors::InvalidReference,"WrongType")
	<< "RefCore: A request to convert a contained product of type '"
	<< TypeID(actualType) << "'\n"
	<< " to type '" << TypeID(expectedType) << "'"
  	<< "\nfor ProductID '" << id()
	<< "' can not be satisfied\n";
  }

  void
  RefCore::nullPointerForTransientException(std::type_info const& type) const {
    throw edm::Exception(errors::InvalidReference)
	<< "RefCore: A request to resolve a transient reference to a product of type: "
	<< TypeID(type)
	<< "\ncan not be satisfied because the pointer to the product is null.\n";
  }

  bool
  RefCore::isAvailable() const {
    ProductID tId = id();
    return productPtr() != 0 || (tId.isValid() && productGetter() != 0 && productGetter()->getIt(tId).isValid());
  }

  void
  RefCore::setProductGetter(EDProductGetter const* prodGetter) const {
    cachePtr_ = prodGetter;
    setCacheIsProductPtr(false);
  }

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
    //Since productPtr and productGetter actually share the same pointer internally,
    // we want to be sure that if the productPtr is set we use that one and only if
    // it isn't set do we set the productGetter if available
    if (productPtr() == 0 && productToBeInserted.productPtr() != 0) {
      setProductPtr(productToBeInserted.productPtr());
    } else if (productPtr() == 0 && productGetter() == 0 && productToBeInserted.productGetter() != 0) {
      setProductGetter(productToBeInserted.productGetter());
    }
  }
  
}
