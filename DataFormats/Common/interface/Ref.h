#ifndef DataFormats_Common_Ref_h
#define DataFormats_Common_Ref_h

/*----------------------------------------------------------------------
  
Ref: A template for a interproduct reference to a member of a product.

$Id: Ref.h,v 1.24 2007/05/16 22:31:59 paterno Exp $

----------------------------------------------------------------------*/
/**
  \b Summary

  The edm::Ref<> is a storable reference to an item in a stored
  container.  For example, you could use one to hold a reference back
  to one particular track within an std::vector<> of tracks.
 
  \b Usage
 
  The edm::Ref<> works just like a pointer
  \code
     edm::Ref<Foo> fooPtr = ... //set the value
     functionTakingConstFoo(*fooPtr); //get the Foo object
     fooPtr->bar();  //call a method of the held Foo object
  \endcode

  The main purpose of an edm::Ref<> is it can be used as a member
  datum for a class that is to be stored in the edm::Event.
 
  \b Customization
 
   The edm::Ref<> takes three template parameters

     1) \b C: The type of the container which is holding the item

     2) \b T: The type of the item.  This defaults to C::value_type

     3) \b F: A helper class (a functor) which knows how to find a
     particular 'T' within the container given an appropriate key. The
     type of the key is deduced from F::second_argument. The default
     for F is refhelper::FindTrait<C,T>::value.  If no specialization
     of FindTrait<> is available for the combination (C,T) then it
     defaults to getting the iterator to be beginning of the container
     and using std::advance() to move to the appropriate key in the
     container.
 
     It is possible to customize the 'lookup' algorithm used.  

     1) The helper class F should inherit from
     std::binary_function<const C&, typename IndexT, const T*> (or
     must provide the typedefs obtained from that inheritance
     directly).

     2) The helper class F must define the function call operator in
     such a way that the following call is well-formed:
         // f    is an instance of type F
         // coll is an instance of type C
         // k    is an instance of type F::key_type

         result_type r = f(coll,k);     
 
     If one wishes to make a specialized lookup the default lookup for
     the container/type pair then one needs to partially specialize
     the templated class edm::refhelper::FindTrait<C,T> such that it
     has a typedef named 'value' which refers to the specialized
     helper class (i.e., F)

     The class template Ref<C,T,F> supports 'null' references.

     -- a default-constructed Ref is 'null'; furthermore, it also
        has an invalid (or 'null') ProductID.
     -- a Ref constructed through the single-arguement constructor
        that takes a ProductID is also null.        
*/

/*----------------------------------------------------------------------
//  This defines the public interface to the class Ref<C, T, F>.
//  C                         is the collection type.
//  T (default C::value_type) is the type of an element in the collection.
//
//  ProductID productID       is the product ID of the collection.
//  key_type itemKey	      is the key of the element in the collection.
//  C::value_type *itemPtr    is a C++ pointer to the element 
//  Ref<C, T, F> const& ref   is another Ref<C, T, F>

//  Constructors
    Ref(); // Default constructor
    Ref(Ref<C, T> const& ref);	// Copy constructor  (default, not explicitly specified)

    Ref(Handle<C> const& handle, key_type itemKey);
    Ref(ProductID pid, key_type itemKey, EDProductGetter const* prodGetter);

//  Destructor
    virtual ~Ref() {}

    // Operators and methods
    Ref<C, T>& operator=(Ref<C, T> const&);		// assignment (default, not explicitly specified)
    T const& operator*() const;			// dereference
    T const* const operator->() const;		// member dereference
    bool operator==(Ref<C, T> const& ref) const; // equality
    bool operator!=(Ref<C, T> const& ref) const; // inequality
    bool operator<(Ref<C, T> const& ref) const; // ordering
    bool isNonnull() const;			// true if an object is referenced
    bool isNull() const;			// equivalent to !isNonnull()
    bool operator!() const;			// equivalent to !isNonnull()
    ----------------------------------------------------------------------*/ 

#include "boost/functional.hpp"
#include "boost/call_traits.hpp"
#include "boost/type_traits.hpp"
#include "boost/mpl/has_xxx.hpp"
#include "boost/utility/enable_if.hpp"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/RefBase.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

BOOST_MPL_HAS_XXX_TRAIT_DEF(key_compare)

#if ! GCC_PREREQUISITE(3,4,4)
// Workaround needed in gcc3.2.3 due to compiler bug
  namespace GCC_3_2_3_WORKAROUND_1 {
#endif
    template <typename C, typename K>
    typename boost::enable_if<has_key_compare<C>, bool>::type
    compare_key(K const& lhs, K const& rhs) {
      typedef typename C::key_compare comparison_functor;
      return comparison_functor()(lhs, rhs);
    }
#if ! GCC_PREREQUISITE(3,4,4)
    // Workaround needed in gcc3.2.3 due to compiler bug
  }
namespace GCC_3_2_3_WORKAROUND_2 {
#endif
  template <typename C, typename K>
  typename boost::disable_if<has_key_compare<C>, bool>::type
  compare_key(K const& lhs, K const& rhs) {
    return lhs < rhs;
  }
#if ! GCC_PREREQUISITE(3,4,4)
  // Workaround needed in gcc3.2.3 due to compiler bug
}
#endif

namespace edm {

  namespace refhelper {
    template<typename C, typename T>
    struct FindUsingAdvance : public std::binary_function<C const&, typename C::size_type, T const*> {
      typedef FindUsingAdvance<C, T> self;
      typename self::result_type operator()(typename self::first_argument_type iContainer,
                                            typename self::second_argument_type iIndex) {
        typename C::const_iterator it = iContainer.begin();
        std::advance(it, iIndex);
        T const* p = it.operator->();
        return p;
      }
    };
    
    //Used in edm::Ref to set the default 'find' method to use based on the Container and 'contained' type
    template<typename C, typename T> 
    struct FindTrait {
      typedef FindUsingAdvance<C, T> value;
    };
  }
  
  template <typename C, typename T = typename C::value_type, class F = typename refhelper::FindTrait<C, T>::value>
  class Ref {
  public:
    friend class RefVector<C, T, F>;
    friend class RefVectorIterator<C, T, F>;

    /// for export
    typedef T value_type; 
    typedef T const element_type; //used for generic programming
    typedef F finder_type;
    typedef typename boost::binary_traits<F>::second_argument_type argument_type;
    typedef typename boost::remove_cv<typename boost::remove_reference<argument_type>::type>::type key_type;   
    /// C is the type of the collection
    /// T is the type of a member the collection

    /// Default constructor needed for reading from persistent store. Not for direct use.
    Ref() : ref_() {}

    /** General purpose constructor from handle like object.
        The templating is artificial.
        HandleC must have the following methods:
        id(), returning a ProductID,
        product(), returning a C*. */
    template <typename HandleC>
    Ref(HandleC const& handle, key_type itemKey, bool setNow=true);

    /** Constructor for those users who do not have a product handle,
        but have a pointer to a product getter (such as the EventPrincipal).
        prodGetter will ususally be a pointer to the event principal. */
    Ref(ProductID const& productID, key_type itemKey, EDProductGetter const* prodGetter) :
      ref_(productID, 0, itemKey, 0, prodGetter) {
      }

    /** Constructor for use in the various X::fillView(...) functions.
	It is an error (not diagnosable at compile- or run-time) to call
	this constructor with a pointer to a T unless the pointed-to T
	object is already in a collection of type C stored in the
	Event. The given ProductID must be the id of the collection in
	the Event. */
    
    Ref(ProductID const& productID, T const* item, key_type item_key) :
      ref_(productID, 0, item_key, item, 0) { }

    /** Constructor that creates an invalid ("null") Ref that is
	associated with a given product (denoted by that product's
	ProductID). */

    explicit Ref(ProductID const& id) :
      ref_(id, 0, key_traits<key_type>::value, 0, 0)
    { }

    /// Constructor from RefProd<C> and key
    Ref(RefProd<C> const& refProd, key_type itemKey);

    /// Destructor
    ~Ref() {}

    /// Dereference operator
    T const&
    operator*() const;

    /// Member dereference operator
    T const*
    operator->() const;

    /// Returns C++ pointer to the item
    T const* get() const {
      return isNull() ? 0 : this->operator->();
    }

    /// Checks for null
    bool isNull() const {return !isNonnull(); }

    /// Checks for non-null
    //bool isNonnull() const {return id().isValid(); }
    bool isNonnull() const { return ref_.isNonnull(); }

    /// Checks for null
    bool operator!() const {return isNull();}

    /// Accessor for product ID.
    ProductID id() const {return ref_.refCore().id();}

    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {return ref_.refCore().productGetter();}

    /// Accessor for product collection
    // Accessor must get the product if necessary
    C const* product() const;

    /// Accessor for product key.
    key_type key() const {return ref_.item().key();}

    // This one just for backward compatibility.  Will be removed soon.
    key_type index() const {return ref_.item().key();}

    /// Accessor for all data
    RefBase<key_type> const& ref() const {return ref_;}

    bool hasProductCache() const {return ref_.refCore().productPtr() != 0;}

    bool hasCache() const {return ref_.item().ptr() != 0;}

  private:
    // Constructor from member of RefVector
    Ref(RefCore const& refCore, RefItem<key_type> const& item) : 
      ref_(refCore, item) {
      }

  private:
    // Compile time check that the argument is a C* or C const*
    // or derived from it.
    void checkTypeAtCompileTime(C const* ptr) {}

    RefBase<key_type> ref_;
  };
}

#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefCoreGet.h"
#include "DataFormats/Common/interface/RefItemGet.h"

namespace edm {
  /** General purpose constructor from handle like object.
      The templating is artificial.
      HandleC must have the following methods:
      id(), returning a ProductID,
      product(), returning a C*. */
  template <typename C, typename T, typename F>
  template <typename HandleC>
  inline
  Ref<C, T, F>::Ref(HandleC const& handle, key_type itemKey, bool setNow) :
      ref_(handle.id(), handle.product(), itemKey) {
    checkTypeAtCompileTime(handle.product());
    assert(ref_.item().key() == itemKey);
        
    if (setNow) {ref_.item().setPtr(getPtr_<C, T, F>(ref_.refCore(), ref_.item()));}
  }

  /// Constructor from RefProd<C> and key
  template <typename C, typename T, typename F>
  inline
  Ref<C, T, F>::Ref(RefProd<C> const& refProd, key_type itemKey) :
      ref_(refProd.id(), refProd.refCore().productPtr(), itemKey, 0, refProd.refCore().productGetter()) {
    assert(ref_.item().key() == itemKey);
    if (0 != refProd.refCore().productPtr()) {
      ref_.item().setPtr(getPtr_<C, T, F>(ref_.refCore(), ref_.item()));
    }
  }

  /// Accessor for product collection
  // Accessor must get the product if necessary
  template <typename C, typename T, typename F>
  inline
  C const*
  Ref<C, T, F>::product() const {
      return isNull() ? 0 : getProduct<C>(ref_.refCore());
  }

  /// Dereference operator
  template <typename C, typename T, typename F>
  inline
  T const&
  Ref<C, T, F>::operator*() const {
    return *getPtr<C, T, F>(ref_.refCore(), ref_.item());
  }

  /// Member dereference operator
  template <typename C, typename T, typename F>
  inline
  T const*
  Ref<C, T, F>::operator->() const {
    return getPtr<C, T, F>(ref_.refCore(), ref_.item());
  }

  template <typename C, typename T, typename F>
  inline
  bool
  operator==(Ref<C, T, F> const& lhs, Ref<C, T, F> const& rhs) {
    return lhs.ref() == rhs.ref();
  }

  template <typename C, typename T, typename F>
  inline
  bool
  operator!=(Ref<C, T, F> const& lhs, Ref<C, T, F> const& rhs) {
    return !(lhs == rhs);
  }

  template <typename C, typename T, typename F>
  inline
  bool
  operator<(Ref<C, T, F> const& lhs, Ref<C, T, F> const& rhs) {
#if ! GCC_PREREQUISITE(3,4,4)
    // needed for gcc 3_2_3 compiler bug workaround
    using GCC_3_2_3_WORKAROUND_1::compare_key;
    using GCC_3_2_3_WORKAROUND_2::compare_key;
#endif
    /// the definition and use of compare_key<> guarantees that the ordering of Refs within
      /// a collection will be identical to the ordering of the referenced objects in the collection.
      return (lhs.id() == rhs.id() ? compare_key<C>(lhs.key(), rhs.key()) : lhs.id() < rhs.id());
  }

}
  
#endif
