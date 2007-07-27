#ifndef DataFormats_Common_RefProd_h
#define DataFormats_Common_RefProd_h

/*----------------------------------------------------------------------
  
Ref: A template for an interproduct reference to a product.

$Id: RefProd.h,v 1.13 2007/07/25 15:33:00 llista Exp $

----------------------------------------------------------------------*/

/*----------------------------------------------------------------------
//  This defines the public interface to the class RefProd<T>.
//
//  ProductID productID		is the product ID of the collection. (0 is invalid)
//  const RefProd<T> & ref		is another RefProd<T>

//  Constructors
    RefProd(); // Default constructor
    RefProd(const RefProd<T> & ref);	// Copy constructor  (default, not explicitly specified)

    RefProd(Handle<T> const& handle);
    RefProd(ProductID pid, EDProductGetter const* prodGetter);

//  Destructor
    virtual ~RefProd() {}

// Operators and methods
    RefProd<T>& operator=(const RefProd<T> &);	// assignment (default, not explicitly specified)
    T const& operator*() const;			// dereference
    T const* operator->() const;		// member dereference
    bool operator==(RefProd<T> const& ref) const;	// equality
    bool operator!=(RefProd<T> const& ref) const;	// inequality
    bool operator<(RefProd<T> const& ref) const;	// ordering
    bool isNonnull() const;			// true if an object is referenced
    bool isNull() const;			// equivalent to !isNonnull()
    bool operator!() const;			// equivalent to !isNonnull()
----------------------------------------------------------------------*/ 

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {

  template <typename C>
  class RefProd {
  public:
    typedef C product_type;
    typedef C value_type;

    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefProd() : product_() {}

    /// General purpose constructor from handle-like object.
    // The templating is artificial.
    // HandleC must have the following methods:
    //   id(),      returning a ProductID,
    //   product(), returning a C*.
    template <class HandleC>
    explicit RefProd(HandleC const& handle) :
    product_(handle.id(), handle.product(), 0) {
      checkTypeAtCompileTime(handle.product());
    }

    /// Constructor from Ref<C,T,F>
    template <typename T, typename F>
    explicit RefProd(Ref<C, T, F> const& ref);

    // Constructor for those users who do not have a product handle,
    // but have a pointer to a product getter (such as the EventPrincipal).
    // prodGetter will ususally be a pointer to the event principal.
    RefProd(ProductID const& productID, EDProductGetter const* prodGetter) :
      product_(productID, 0, prodGetter) {
    }

    /// Destructor
    ~RefProd() {}

    /// Dereference operator
    product_type const&  operator*() const;

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
    bool isNull() const {return !isNonnull(); }

    /// Checks for non-null
    bool isNonnull() const {return id().isValid(); }

    /// Checks for null
    bool operator!() const {return isNull(); }

    /// Accessor for product ID.
    ProductID id() const {return product_.id();}

    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {return product_.productGetter();}

    /// Checks if product is in memory.
    bool hasCache() const {return product_.productPtr() != 0;}

    void swap( RefProd<C> & );

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

  /// Constructor from Ref.
  template <typename C>
  template <typename T, typename F>
  inline
  RefProd<C>::RefProd(Ref<C, T, F> const& ref) :
      product_(ref.id(), 
	       ref.hasProductCache() ? 
	       ref.product() : 
	       0, ref.productGetter()) 
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
  void RefProd<C>::swap( RefProd<C> & other ) {
    std::swap( product_, other.product_ );
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
  void swap( const edm::RefProd<C> & lhs, const edm::RefProd<C> & rhs ) {
    lhs.swap( rhs );
  }
}

#include "DataFormats/Common/interface/HolderToVectorTrait.h"

namespace edm {
  namespace reftobase {

    template <typename T>
    struct RefProdHolderToVector {
      static  std::auto_ptr<BaseVectorHolder<T> > makeVectorHolder() {
	throw edm::Exception(errors::InvalidReference)
	  << "attempting to make a BaseVectorHolder<T> from a RefProd<C>.";
      }
    };

    template<typename C, typename T>
    struct HolderToVectorTrait<T, RefProd<C> > {
      typedef RefProdHolderToVector<T> type;
    };

    struct RefProdRefHolderToRefVector {
      static  std::auto_ptr<RefVectorHolderBase> makeVectorHolder() {
	throw edm::Exception(errors::InvalidReference)
	  << "attempting to make a RefVectorHolderBase from a RefProd<C>.";
      }
    };

    template<typename C>
    struct RefHolderToRefVectorTrait<RefProd<C> > {
      typedef RefProdRefHolderToRefVector type;
    };

  }
}

#endif
