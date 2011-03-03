#ifndef DataFormats_Common_RefProd_h
#define DataFormats_Common_RefProd_h

/*----------------------------------------------------------------------
  
Ref: A template for an interproduct reference to a product.

$Id: RefProd.h,v 1.20 2009/11/04 15:30:20 wmtan Exp $

----------------------------------------------------------------------*/

/*----------------------------------------------------------------------
//  This defines the public interface to the class RefProd<T>.
//
//  ProductID productID		is the product ID of the collection. (0 is invalid)
//  RefProd<T> const& ref	is another RefProd<T>

//  Constructors
    RefProd(); // Default constructor
    RefProd(RefProd<T> const& ref);	// Copy constructor  (default, not explicitly specified)

    RefProd(Handle<T> const& handle);
    RefProd(ProductID pid, EDProductGetter const* prodGetter);

//  Destructor
    virtual ~RefProd() {}

// Operators and methods
    RefProd<T>& operator=(RefProd<T> const&);	// assignment (default, not explicitly specified)
    T const& operator*() const;			// dereference
    T const* operator->() const;		// member dereference
    bool operator==(RefProd<T> const& ref) const;	// equality
    bool operator!=(RefProd<T> const& ref) const;	// inequality
    bool operator<(RefProd<T> const& ref) const;	// ordering
    bool isNonnull() const;			// true if an object is referenced
    bool isNull() const;			// equivalent to !isNonnull()
    bool operator!() const;			// equivalent to !isNonnull()
----------------------------------------------------------------------*/ 

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/TestHandle.h"

namespace edm {

  template <typename C>
  class RefProd {
  public:
    typedef C product_type;
    typedef C value_type;

    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefProd() : product_() {}

    /// General purpose constructor from handle.
    explicit RefProd(Handle<C> const& handle) :
    product_(handle.id(), handle.product(), 0, false) {
      checkTypeAtCompileTime(handle.product());
    }

    /// General purpose constructor from orphan handle.
    explicit RefProd(OrphanHandle<C> const& handle) :
    product_(handle.id(), handle.product(), 0, false) {
      checkTypeAtCompileTime(handle.product());
    }

    /// Constructor for ref to object that is not in an event.
    //  An exception will be thrown if an attempt is made to persistify
    //  any object containing this RefProd.  Also, in the future work will
    //  be done to throw an exception if an attempt is made to put any object
    //  containing this RefProd into an event(or run or lumi).
    RefProd(C const* product) :
      product_(ProductID(), product, 0, true) {
      checkTypeAtCompileTime(product);
    }

    /// General purpose constructor from test handle.
    //  An exception will be thrown if an attempt is made to persistify
    //  any object containing this RefProd.  Also, in the future work will
    //  be done to throw an exception if an attempt is made to put any object
    //  containing this RefProd into an event(or run or lumi).
    explicit RefProd(TestHandle<C> const& handle) :
    product_(handle.id(), handle.product(), 0, true) {
      checkTypeAtCompileTime(handle.product());
    }

    /// Constructor from Ref<C,T,F>
    template <typename T, typename F>
    explicit RefProd(Ref<C, T, F> const& ref);

    /// Constructor from RefVector<C,T,F>
    template <typename T, typename F>
    explicit RefProd(RefVector<C, T, F> const& ref);

    // Constructor for those users who do not have a product handle,
    // but have a pointer to a product getter (such as the EventPrincipal).
    // prodGetter will ususally be a pointer to the event principal.
    RefProd(ProductID const& productID, EDProductGetter const* prodGetter) :
      product_(productID, 0, mustBeNonZero(prodGetter, "RefProd", productID), false) {
    }

    /// Destructor
    ~RefProd() {}

    /// Dereference operator
    product_type const& operator*() const;

    /// Member dereference operator
    product_type const* operator->() const;

    /// Returns C++ pointer to the product
    /// Will attempt to retrieve product
    product_type const* get() const {
      return isNull() ? 0 : this->operator->();
    }

    /// Returns C++ pointer to the product
    /// Will attempt to retrieve product
    product_type const* product() const {
      return isNull() ? 0 : this->operator->();
    }

    RefCore const& refCore() const {
      return product_;
    }

    /// Checks for null
    bool isNull() const {return !isNonnull();}

    /// Checks for non-null
    bool isNonnull() const {return product_.isNonnull();}

    /// Checks for null
    bool operator!() const {return isNull();}

    /// Accessor for product ID.
    ProductID id() const {return product_.id();}

    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {return product_.productGetter();}

    /// Checks if product is in memory.
    bool hasCache() const {return product_.productPtr() != 0;}

    /// Checks if product is in memory.
    bool hasProductCache() const {return hasCache();}

    /// Checks if collection is in memory or available
    /// in the Event. No type checking is done.
    bool isAvailable() const {return product_.isAvailable();}

    /// Checks if this RefProd is transient (i.e. not persistable).
    bool isTransient() const {return product_.isTransient();}

    void swap(RefProd<C> &);

    //Needed for ROOT storage
    CMS_CLASS_VERSION(10)

  private:
    // Compile time check that the argument is a C* or C const*
    // or derived from it.
    void checkTypeAtCompileTime(C const* ptr) {}

    RefCore product_;
  };
}

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefCoreGet.h"

namespace edm {
  template<typename C, typename T, typename F>
  class RefVector;

  /// Constructor from Ref.
  template <typename C>
  template <typename T, typename F>
  inline
  RefProd<C>::RefProd(Ref<C, T, F> const& ref) :
      product_(ref.id(), ref.hasProductCache() ?  ref.product() : 0, ref.productGetter(), ref.isTransient()) 
  {  }

  /// Constructor from RefVector.
  template <typename C>
  template <typename T, typename F>
  inline
  RefProd<C>::RefProd(RefVector<C, T, F> const& ref) :
      product_(ref.id(), ref.hasProductCache() ?  ref.product() : 0, ref.productGetter(), ref.isTransient()) 
  {  }

  /// Dereference operator
  template <typename C>
  inline
  C const& RefProd<C>::operator*() const {
    return *(edm::template getProduct<C>(product_));
  }

  /// Member dereference operator
  template <typename C>
  inline
  C const* RefProd<C>::operator->() const {
    return edm::template getProduct<C>(product_);
  } 


  template<typename C>
  inline
  void RefProd<C>::swap(RefProd<C> & other) {
    std::swap(product_, other.product_);
  }

  template <typename C>
  inline
  bool
  operator== (RefProd<C> const& lhs, RefProd<C> const& rhs) {
    return lhs.refCore() == rhs.refCore();
  }

  template <typename C>
  inline
  bool
  operator!= (RefProd<C> const& lhs, RefProd<C> const& rhs) {
    return !(lhs == rhs);
  }

  template <typename C>
  inline
  bool
  operator< (RefProd<C> const& lhs, RefProd<C> const& rhs) {
    return (lhs.refCore() < rhs.refCore());
  }

  template<typename C>
  inline
  void swap(edm::RefProd<C> const& lhs, edm::RefProd<C> const& rhs ) {
    lhs.swap(rhs);
  }
}

#include "DataFormats/Common/interface/HolderToVectorTrait.h"

namespace edm {
  namespace reftobase {

    template <typename T>
    struct RefProdHolderToVector {
      static  std::auto_ptr<BaseVectorHolder<T> > makeVectorHolder() {
	Exception::throwThis(errors::InvalidReference, "attempting to make a BaseVectorHolder<T> from a RefProd<C>.\n");
	return std::auto_ptr<BaseVectorHolder<T> >();
      }
      static std::auto_ptr<RefVectorHolderBase> makeVectorBaseHolder() {
	Exception::throwThis(errors::InvalidReference, "attempting to make a RefVectorHolderBase from a RefProd<C>.\n");
	return std::auto_ptr<RefVectorHolderBase>();
      }
    };

    template<typename C, typename T>
    struct HolderToVectorTrait<T, RefProd<C> > {
      typedef RefProdHolderToVector<T> type;
    };

    struct RefProdRefHolderToRefVector {
      static  std::auto_ptr<RefVectorHolderBase> makeVectorHolder() {
	Exception::throwThis(errors::InvalidReference, "attempting to make a BaseVectorHolder<T> from a RefProd<C>.\n");
	return std::auto_ptr<RefVectorHolderBase>();
      }
      static  std::auto_ptr<RefVectorHolderBase> makeVectorBaseHolder() {
	Exception::throwThis(errors::InvalidReference, "attempting to make a RefVectorHolderBase from a RefProd<C>.\n");
	return std::auto_ptr<RefVectorHolderBase>();
      }
    };

    template<typename C>
    struct RefHolderToRefVectorTrait<RefProd<C> > {
      typedef RefProdRefHolderToRefVector type;
    };

  }
}

#endif
