#ifndef DataFormats_Common_FwdRef_h
#define DataFormats_Common_FwdRef_h

/*----------------------------------------------------------------------
  
FwdRef: A template for a interproduct reference to a member of a product.

----------------------------------------------------------------------*/
/**
  \b Summary

  The edm::FwdRef<> is a storable reference to an item in a stored
  "forward" container, which also contains a reference to an item in a
  "backward" container that the "forward" container is derived from.

  For example, you could use one to hold a reference back
  to one particular track within a derived std::vector<> of tracks, 
  but you want to keep the original Ref's to the original
  std::vector<> of tracks (for instance, if you've made a selection
  on the tracks in the list and want to remove the unnecessary ones
  from the event). 
 
  \b Usage
 
  The edm::FwdRef<> works just like a pointer
  \code
     edm::FwdRef<Foo> fooPtr = ... //set the value
     functionTakingConstFoo(*fooPtr); //get the Foo object
     fooPtr->bar();  //call a method of the held Foo object
  \endcode

  The main purpose of an edm::FwdRef<> is it can be used as a member
  datum for a class that is to be stored in the edm::Event where the
  user can simultaneously check the "backwards" ref as well as the
  default "forward" ref. 
 
  \b Customization
 
   The edm::FwdRef<> takes three template parameters, and both
   "forward" and "backward" refs must be the same types:

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

     The class template FwdRef<C,T,F> supports 'null' references.

     -- a default-constructed FwdRef is 'null'; furthermore, it also
        has an invalid (or 'null') ProductID.
     -- a FwdRef constructed through the single-arguement constructor
        that takes a ProductID is also null.        
*/

/*----------------------------------------------------------------------
//  This defines the public interface to the class FwdRef<C, T, F>.
//  C                         is the collection type.
//  T (default C::value_type) is the type of an element in the collection.
//
//  ProductID productID       is the product ID of the collection.
//  key_type itemKey	      is the key of the element in the collection.
//  C::value_type *itemPtr    is a C++ pointer to the element 
//  FwdRef<C, T, F> const& ref   is another FwdRef<C, T, F>

//  Constructors
    FwdRef(); // Default constructor
    FwdRef(FwdRef<C, T> const& ref);	// Copy constructor  (default, not explicitly specified)

    FwdRef(Ref<C,T,F> const & fwdRef, Ref<C,T,F> const & backRef);

//  Destructor
    virtual ~FwdRef() {}

    // Operators and methods
    FwdRef<C, T>& operator=(FwdRef<C, T> const&);		// assignment (default, not explicitly specified)
    T const& operator*() const;			// dereference
    T const* const operator->() const;		// member dereference
    bool operator==(FwdRef<C, T> const& ref) const; // equality
    bool operator!=(FwdRef<C, T> const& ref) const; // inequality
    bool operator<(FwdRef<C, T> const& ref) const; // ordering
    bool isNonnull() const;			// true if an object is referenced
    bool isNull() const;			// equivalent to !isNonnull()
    bool operator!() const;			// equivalent to !isNonnull()
    ----------------------------------------------------------------------*/ 

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/Ref.h"

namespace edm {

  template <typename C, 
	    typename T = typename refhelper::ValueTrait<C>::value, 
	    typename F = typename refhelper::FindTrait<C, T>::value>
  class FwdRef {


  public:
    /// for export
    typedef C product_type;
    typedef T value_type; 
    typedef T const element_type; //used for generic programming
    typedef F finder_type;
    typedef typename boost::binary_traits<F>::second_argument_type argument_type;
    typedef typename boost::remove_cv<typename boost::remove_reference<argument_type>::type>::type key_type;   
    /// C is the type of the collection
    /// T is the type of a member the collection

    /// Default constructor needed for reading from persistent store. Not for direct use.
    FwdRef() : ref_(), backRef_() {}

    /// General purpose constructor from 2 refs (forward and backward.
    FwdRef(Ref<C,T,F> const & ref,
	   Ref<C,T,F> const & backRef) :
        ref_(ref), backRef_(backRef) {}
  
    /// Destructor
    ~FwdRef() {}

    /// Dereference operator
    T const&
    operator*() const;

    /// Member dereference operator
    T const*
    operator->() const;

    /// Returns C++ pointer to the item
    T const* get() const {
      if ( ref_.isNonnull() ) {
	return ref_.get();
      } else if ( backRef_.isNonnull() ) {
	return backRef_.get();
      } else {
	return 0;
      }
    }

    /// Checks for null
    bool isNull() const {return !isNonnull(); }

    /// Checks for non-null
    //bool isNonnull() const {return id().isValid(); }
    bool isNonnull() const { return ref_.isNonnull() || backRef_.isNonnull(); }

    /// Checks for null
    bool operator!() const {return isNull();}

    Ref<C,T,F> const & ref() const { return ref_; }
    Ref<C,T,F> const & backRef() const { return backRef_; }

    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {
      if ( ref_.productGetter() ) return ref_.productGetter();
      else return backRef_.productGetter();
    }

    /// Accessor for product collection
    // Accessor must get the product if necessary
    C const* product() const;

    /// Accessor for product ID.
    ProductID id() const {return ref_.isNonnull() ? ref_.id() : backRef_.id();}


    /// Accessor for product key.
    key_type key() const {return ref_.isNonnull() ? ref_.key() : backRef_.key() ;}

    bool hasProductCache() const {return ref_.hasProductCache() || backRef_.hasProductCache();}

    bool hasCache() const {return ref_.hasCache() || backRef_.hasCache();}

    /// Checks if collection is in memory or available
    /// in the Event. No type checking is done.
    bool isAvailable() const {return ref_.isAvailable() || backRef_.isAvailable();}

    /// Checks if this ref is transient (i.e. not persistable).
    bool isTransient() const {return ref_.isTransient();}

    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:
    Ref<C,T,F> ref_;
    Ref<C,T,F> backRef_;
  };
}

#include "DataFormats/Common/interface/RefProd.h"

namespace edm {


  /// Accessor for product collection
  // Accessor must get the product if necessary
  template <typename C, typename T, typename F>
  inline
  C const*
  FwdRef<C, T, F>::product() const {
    return ref_.isNonnull() && ref_.isAvailable() ? 
      ref_.product() :
      backRef_.product();
  }

  /// Dereference operator
  template <typename C, typename T, typename F>
  inline
  T const&
  FwdRef<C, T, F>::operator*() const {
    return ref_.isNonnull() && ref_.isAvailable() ?
      ref_.operator*() :
      backRef_.operator*();
  }

  /// Member dereference operator
  template <typename C, typename T, typename F>
  inline
  T const*
  FwdRef<C, T, F>::operator->() const {
    return ref_.isNonnull() && ref_.isAvailable() ?
      ref_.operator->() :
      backRef_.operator->();
  }


  /// for two FwdRefs to be equal, both the 
  /// "forward" and the "backward" Refs must be the same
  template <typename C, typename T, typename F>
  inline
  bool
  operator==(FwdRef<C, T, F> const& lhs, FwdRef<C, T, F> const& rhs) {
    return 
    (lhs.ref() == rhs.ref() ) &&
    (lhs.backRef() == rhs.backRef() )
    ;
  }

  /// for a FwdRef to equal a Ref, EITHER the
  /// "forward" or the "backward" Refs must equal to the test ref
  template <typename C, typename T, typename F>
  inline
  bool
  operator==(Ref<C, T, F> const& lhs, FwdRef<C, T, F> const& rhs) {
    return 
    (lhs == rhs.ref() )    ||
    (lhs == rhs.backRef() )
    ;
  }

  /// for a FwdRef to equal a Ref, EITHER the
  /// "forward" or the "backward" Refs must equal to the test ref
  template <typename C, typename T, typename F>
  inline
  bool
  operator==(FwdRef<C, T, F> const& lhs, Ref<C, T, F> const& rhs) {
    return 
    (lhs.ref() == rhs )    ||
    (lhs.backRef() == rhs )
    ;
  }



  template <typename C, typename T, typename F>
  inline
  bool
  operator!=(FwdRef<C, T, F> const& lhs, FwdRef<C, T, F> const& rhs) {
    return !(lhs == rhs);
  }

  template <typename C, typename T, typename F>
  inline
  bool
  operator!=(Ref<C, T, F> const& lhs, FwdRef<C, T, F> const& rhs) {
    return !(lhs == rhs);
  }

  template <typename C, typename T, typename F>
  inline
  bool
  operator!=(FwdRef<C, T, F> const& lhs, Ref<C, T, F> const& rhs) {
    return !(lhs == rhs);
  }

  /// for inequality operators, ONLY test the forward ref.
  /// the ordering of the backward ref is not relevant.
  template <typename C, typename T, typename F>
  inline
  bool
  operator<(FwdRef<C, T, F> const& lhs, FwdRef<C, T, F> const& rhs) {
    return (lhs.ref() < rhs.ref() );
  }

  template <typename C, typename T, typename F>
  inline
  bool
  operator<(Ref<C, T, F> const& lhs, FwdRef<C, T, F> const& rhs) {
    return (lhs < rhs.ref() );
  }

  template <typename C, typename T, typename F>
  inline
  bool
  operator<(FwdRef<C, T, F> const& lhs, Ref<C, T, F> const& rhs) {
    return (lhs.ref() < rhs );
  }

}

  
#endif
