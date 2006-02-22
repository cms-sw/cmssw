#ifndef Common_Ref_h
#define Common_Ref_h

/*----------------------------------------------------------------------
  
Ref: A template for a interproduct reference to a member of a product.

$Id: Ref.h,v 1.1 2006/02/07 07:01:50 wmtan Exp $

----------------------------------------------------------------------*/

/*----------------------------------------------------------------------
//  This defines the public interface to the class Ref<C, T>.
//  C				is the collection type.
//  T (default C::value_type)	is the type of an object inthe collection.
//
//  ProductID productID		is the product ID of the collection. (0 is invalid)
//  size_type itemIndex		is the index of the object into the collection.
//  C::value_type *itemPtr	is a C++ pointer to the object in memory.
//  const Ref<C, T> & ref	is another Ref<C, T>

//  Constructors
    Ref(); // Default constructor
    Ref(const Ref<C, T> & ref);	// Copy constructor  (default, not explicitly specified)

    Ref(Handle<C> const& handle, size_type itemIndex);
    Ref(ProductID pid, size_type itemIndex, EDProductGetter const* prodGetter);

//  Destructor
    virtual ~Ref() {}

// Operators and methods
    Ref<C, T>& operator=(const Ref<C, T> &);		// assignment (default, not explicitly specified)
    T const& operator*() const;			// dereference
    T const* const operator->() const;		// member dereference
    bool operator==(const Ref<C, T>& ref) const; // equality
    bool operator!=(const Ref<C, T>& ref) const; // inequality
    bool isNonnull() const;			// true if an object is referenced
    bool isNull() const;			// equivalent to !isNonnull()
    operator bool() const;			// equivalent to isNonnull()
    bool operator!() const;			// equivalent to !isNonnull()
----------------------------------------------------------------------*/ 

#include <stdexcept>
#include <iterator>
#include <typeinfo>
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/RefBase.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/ProductID.h"

namespace edm {
  template<typename C, typename T> class RefVector;
  template<typename C, typename T> class RefVectorIterator;
  template <typename C, typename T = typename C::value_type>
  class Ref {
  public:
    friend class RefVector<C, T>;
    friend class RefVectorIterator<C, T>;

    /// for export
    typedef T value_type;
    
    /// C is the type of the collection
    /// T is the type of a member the collection

    typedef RefItem::size_type size_type;

    /// Default constructor needed for reading from persistent store. Not for direct use.
    Ref() : ref_() {}

    /// General purpose constructor from handle like object.
    // The templating is artificial.
    // HandleC must have the following methods:
    // id(), returning a ProductID,
    // product(), returning a C*.
    template <typename HandleC>
      Ref(HandleC const& handle, size_type itemIndex) :
      ref_(handle.id(), handle.product(), itemIndex) {
        assert(ref_.item().index() == itemIndex);
        ref_.item().setPtr(getPtr_<C, T>(ref_.product(), ref_.item()));
    }

    // Constructor for those users who do not have a product handle,
    // but have a pointer to a product getter (such as the EventPrincipal).
    // prodGetter will ususally be a pointer to the event principal.
    Ref(ProductID const& productID, size_type itemIndex, EDProductGetter const* prodGetter) :
        ref_(productID, 0, itemIndex, 0, prodGetter) {
    }

    // Constructor from RefProd<C> and index
    Ref(RefProd<C> const& refProd, size_type itemIndex) :
      ref_(refProd.id(), refProd.product().productPtr(), itemIndex, 0, refProd.product().productGetter()) {
        assert(ref_.item().index() == itemIndex);
        ref_.item().setPtr(getPtr_<C, T>(ref_.product(), ref_.item()));
    }

    /// Destructor
    ~Ref() {}

    /// Dereference operator
    T const&
    operator*() const {
      return *getPtr<C, T>(ref_.product(), ref_.item());
    }

    /// Member dereference operator
    T const *
    operator->() const {
      return getPtr<C, T>(ref_.product(), ref_.item());
    }

    /// Checks for null
    bool isNull() const {return id() == ProductID();}

    /// Checks for non-null
    bool isNonnull() const {return !isNull();}

    /// Checks for null
    bool operator!() const {return isNull();}

    /// Checks for non-null
    operator bool() const {return !isNull();}

    /// Accessor for product ID.
    ProductID id() const {return ref_.product().id();}

    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {return ref_.product().productGetter();}

    /// Accessor for product collection
    C const* product() const {return static_cast<C const *>(ref_.product().productPtr());}

    /// Accessor for product index.
    size_type index() const {return ref_.item().index();}

    /// Accessor for all data
    RefBase const& ref() const {return ref_;}

  private:
    // Constructor from member of RefVector
    Ref(RefCore const& product, RefItem const& item) : 
      ref_(product, item) {
    }

  private:
    RefBase ref_;
  };

  template <typename C, typename T>
  inline
  bool
  operator==(Ref<C, T> const& lhs, Ref<C, T> const& rhs) {
    return lhs.ref() == rhs.ref();
  }

  template <typename C, typename T>
  inline
  bool
  operator!=(Ref<C, T> const& lhs, Ref<C, T> const& rhs) {
    return !(lhs == rhs);
  }

}
  
#endif
