#ifndef Common_Ref_h
#define Common_Ref_h

/*----------------------------------------------------------------------
  
Ref: A template for a interproduct reference to a member of a product.

$Id: Ref.h,v 1.7 2006/06/02 01:58:15 wmtan Exp $

----------------------------------------------------------------------*/
/**
  \b Summary
  The edm::Ref<> is a storable reference to an item in a stored container.  
  For example, you could use one to hold a reference back to one particular track
  within an std::vector<> of tracks.
 
  \b Usage
 
  The edm::Ref<> works just like a pointer
  \code
     edm::Ref<Foo> fooPtr = ... //set the value
     functionTakingConstFoo( *fooPtr ); //get the Foo object
     fooPtr->bar();  //call a method of the held Foo object
  \endcode

  The main purpose of an edm::Ref<> is it can be used as a member datum for
  a class that is to be stored in the edm::Event.
 
  \b Customization
 
   The edm::Ref<> takes three template parameters
     1) \b C: The type of the container which is holding the item
     2) \b T: The type of the item.  This defaults to C::value_type
     3) \b F: A helper class (a functor) which knows how to find a particular 'T' within the container
          given an appropriate key. The type of the key is deduced from F::second_argument. 
          The default for F is refhelper::FindTrait<C,T>::value.  If no specialization of FindTrait<> is
          available for the conbination (C,T) then it defaults to getting the iterator to be beginning of
          the container and using std::advance() to move to the appropriate key in the container.
 
     It is possible to customize the 'lookup' algorithm used.  The helper class should inherit from 
     std::binary_function<const C&, typename IndexT, const T*>
 
     and define the function
     result_type operator()( first_argument_type iContainer, second_argument_type iIndex)
 
     where result_type, first_argument_type and second_argument_type are typedefs inherited from std::binary_function<>
 
     If one wishes to make a specialized lookup the default lookup for the container/type pair then one
     needs to partially specialize the templated class edm::refhelper::FindTrait<C,T> such that it has a 
     typedef named 'value' which refers to the specialized helper class (i.e., F)
*/
/*----------------------------------------------------------------------
//  This defines the public interface to the class Ref<C, T>.
//  C				is the collection type.
//  T (default C::value_type)	is the type of an object inthe collection.
//
//  ProductID productID		is the product ID of the collection. (0 is invalid)
//  key_type itemKey		is the key of the object in the collection.
//  C::value_type *itemPtr	is a C++ pointer to the object in memory.
//  Ref<C, T> const& ref	is another Ref<C, T>

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
#include "DataFormats/Common/interface/RefBase.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/ProductID.h"

#if 0
// try if this works with gcc 3.4.3.
#include "boost/mpl/has_xxx.hpp"
#include "boost/utility/enable_if.hpp"
BOOST_MPL_HAS_XXX_TRAIT_DEF(key_compare);

template <typename C, typename K>
typename boost::enable_if<has_key_compare<C>, bool>::type
compare_key(K const& lhs, K const& rhs) {
  return C::key_compare()(lhs, rhs);
}

template <typename C, typename K>
typename boost::disable_if<has_key_compare<C>, bool>::type
compare_key(K const& lhs, K const& rhs) {
  return lhs < rhs;
}
#endif

namespace edm {
  template<typename C, typename T, typename F> class RefVector;
  template<typename C, typename T, typename F> class RefVectorIterator;
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
      Ref(HandleC const& handle, key_type itemKey, bool setNow=true) :
      ref_(handle.id(), handle.product(), itemKey) {
        assert(ref_.item().key() == itemKey);
        if(setNow) {ref_.item().setPtr(getPtr_<C, T, F>(ref_.product(), ref_.item()));}
    }

    /** Constructor for those users who do not have a product handle,
        but have a pointer to a product getter (such as the EventPrincipal).
        prodGetter will ususally be a pointer to the event principal. */
    Ref(ProductID const& productID, key_type itemKey, EDProductGetter const* prodGetter) :
        ref_(productID, 0, itemKey, 0, prodGetter) {
    }

    /// Constructor from RefProd<C> and key
    Ref(RefProd<C> const& refProd, key_type itemKey) :
      ref_(refProd.id(), refProd.product().productPtr(), itemKey, 0, refProd.product().productGetter()) {
        assert(ref_.item().key() == itemKey);
        if(0!=refProd.product().productPtr()) {
          ref_.item().setPtr(getPtr_<C, T, F>(ref_.product(), ref_.item()));
        }
    }

    /// Destructor
    ~Ref() {}

    /// Dereference operator
    T const&
    operator*() const {
      return *getPtr<C, T, F>(ref_.product(), ref_.item());
    }

    /// Member dereference operator
    T const*
    operator->() const {
      return getPtr<C, T, F>(ref_.product(), ref_.item());
    }

    T const* get() const {
      return this->operator->();
    }
    /// Checks for null
    bool isNull() const {return id() == ProductID();}

    /// Checks for non-null
    bool isNonnull() const {return !isNull();}

    /// Checks for null
    bool operator!() const {return isNull();}

    /// Accessor for product ID.
    ProductID id() const {return ref_.product().id();}

    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {return ref_.product().productGetter();}

    /// Accessor for product collection
    C const* product() const {return static_cast<C const *>(ref_.product().productPtr());}

    /// Accessor for product key.
    key_type key() const {return ref_.item().key();}

    // This one just for backward compatibility.  Will be removed soon.
    key_type index() const {return ref_.item().key();}

    /// Accessor for all data
    RefBase<key_type> const& ref() const {return ref_;}

  private:
    // Constructor from member of RefVector
    Ref(RefCore const& product, RefItem<key_type> const& item) : 
      ref_(product, item) {
    }

  private:
    RefBase<key_type> ref_;
  };

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
#if 0
   // try if this works with gcc 3.4.3.
    return (lhs.id() == rhs.id() ? compare_key<C>(lhs.key(), rhs.key()) : lhs.id() < rhs.id());
#else
    return (lhs.id() == rhs.id() ? lhs.key() < rhs.key() : lhs.id() < rhs.id());
#endif
  }

}
  
#endif
