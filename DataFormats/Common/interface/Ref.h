#ifndef DataFormats_Common_Ref_h
#define DataFormats_Common_Ref_h

/*----------------------------------------------------------------------
  
Ref: A template for a interproduct reference to a member of a product_.

----------------------------------------------------------------------*/
/**
  \b Summary

  The edm::Ref<> is a storable reference to an item in a stored
  container.  For example, you could use one to hold a reference back
  to one particular track within an std::vector<> of tracks.
 
  \b Usage
 
  The edm::Ref<> works just like a pointer
  \code
     edm::Ref<FooCollection> fooPtr = ... //set the value
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
     for F is refhelper::FindTrait<C, T>::value.  If no specialization
     of FindTrait<> is available for the combination (C, T) then it
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

         result_type r = f(coll, k);     
 
     If one wishes to make a specialized lookup the default lookup for
     the container/type pair then one needs to partially specialize
     the templated class edm::refhelper::FindTrait<C, T> such that it
     has a typedef named 'value' which refers to the specialized
     helper class (i.e., F)

     The class template Ref<C, T, F> supports 'null' references.

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

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefCoreWithIndex.h"
#include "DataFormats/Common/interface/TestHandle.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "boost/functional.hpp"
#include "boost/call_traits.hpp"
#include "boost/type_traits.hpp"
#include "boost/mpl/has_xxx.hpp"
#include "boost/utility/enable_if.hpp"

#include <vector>

BOOST_MPL_HAS_XXX_TRAIT_DEF(key_compare)

  template <typename C, typename K>
  typename boost::enable_if<has_key_compare<C>, bool>::type
  compare_key(K const& lhs, K const& rhs) {
    typedef typename C::key_compare comparison_functor;
    return comparison_functor()(lhs, rhs);
  }

  template <typename C, typename K>
  typename boost::disable_if<has_key_compare<C>, bool>::type
  compare_key(K const& lhs, K const& rhs) {
    return lhs < rhs;
  }

#include "DataFormats/Common/interface/RefTraits.h"

namespace edm {
  template<typename C, typename T, typename F>
  class RefVector;

  template<typename T>
  class RefToBaseVector;

  template <typename C, 
	    typename T = typename refhelper::ValueTrait<C>::value, 
	    typename F = typename refhelper::FindTrait<C, T>::value>
  class Ref {
  private:
    typedef refhelper::FindRefVectorUsingAdvance<RefVector<C, T, F> > VF;
    typedef refhelper::FindRefVectorUsingAdvance<RefToBaseVector<T> > VBF;
    friend class RefVectorIterator<C, T, F>;
    friend class RefVector<C, T, F>;
    friend class RefVector<RefVector<C, T, F>, T, VF>;
    friend class RefVector<RefVector<C, T, F>, T, VBF>;

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

    static key_type invalidKey() { return key_traits<key_type>::value; }

    /// Default constructor needed for reading from persistent store. Not for direct use.
    Ref() : product_(), index_(key_traits<key_type>::value) {}

    /// General purpose constructor from handle.
    Ref(Handle<C> const& handle, key_type itemKey, bool setNow=true);

    /// General purpose constructor from orphan handle.
    Ref(OrphanHandle<C> const& handle, key_type itemKey, bool setNow=true);

    /// Constructors for ref to object that is not in an event.
    //  An exception will be thrown if an attempt is made to persistify
    //  any object containing this Ref.  Also, in the future work will
    //  be done to throw an exception if an attempt is made to put any object
    //  containing this Ref into an event(or run or lumi).
    Ref(C const* product, key_type itemKey, bool setNow=true);

    /// Constructor from test handle.
    //  An exception will be thrown if an attempt is made to persistify
    //  any object containing this Ref.  Also, in the future work will
    Ref(TestHandle<C> const& handle, key_type itemKey, bool setNow=true);

    /// Constructor for those users who do not have a product handle,
    /// but have a pointer to a product getter (such as the EventPrincipal).
    /// prodGetter will ususally be a pointer to the event principal.
    Ref(ProductID const& productID, key_type itemKey, EDProductGetter const* prodGetter) :
      product_(productID, 0, mustBeNonZero(prodGetter, "Ref", productID), false), index_(itemKey) {
    }

    /// Constructor for use in the various X::fillView(...) functions.
    //  It is an error (not diagnosable at compile- or run-time) to call
    //  this constructor with a pointer to a T unless the pointed-to T
    //  object is already in a collection of type C stored in the
    //  Event. The given ProductID must be the id of the collection in
    //  the Event.
    
    Ref(ProductID const& iProductID, T const* item, key_type itemKey, C const* /* iProduct */) :
      product_(iProductID, item, 0, false), index_(itemKey)
    { }

    Ref(ProductID const& iProductID, T const* item, key_type itemKey) :
      product_(iProductID, item, 0, false), index_(itemKey)
    { }

    Ref(ProductID const& iProductID, T const* item, key_type itemKey, bool transient) :
      product_(iProductID, item, 0, transient), index_(itemKey)
    { }

    /// Constructor that creates an invalid ("null") Ref that is
    /// associated with a given product (denoted by that product's
    /// ProductID).

    explicit Ref(ProductID const& iId) :
      product_(iId, 0, 0, false), index_(key_traits<key_type>::value)
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
    bool isNonnull() const { return index_!=edm::key_traits<key_type>::value; }

    /// Checks for null
    bool operator!() const {return isNull();}

    /// Accessor for product ID.
    ProductID id() const {return product_.id();}

    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {return product_.productGetter();}

    /// Accessor for product key.
    key_type key() const {return index_;}

    // This one just for backward compatibility.  Will be removed soon.
    key_type index() const {return index_;}

    /// Returns true if container referenced by the Ref has been cached
    bool hasProductCache() const {return product_.productPtr() != 0;}

    /// Checks if collection is in memory or available
    /// in the Event. No type checking is done.
    bool isAvailable() const;

    /// Checks if this ref is transient (i.e. not persistable).
    bool isTransient() const {return product_.isTransient();}

    RefCore const& refCore() const {return product_;}

    //Used by ROOT storage
    CMS_CLASS_VERSION(10)
    //  private:
    // Constructor from member of RefVector
    Ref(RefCore const& iRefCore, key_type const& iKey) : 
      product_(iRefCore), index_(iKey) {
    }

  private:

    // Compile time check that the argument is a C* or C const*
    // or derived from it.
    void checkTypeAtCompileTime(C const*) {}

    RefCore product_;
    key_type index_;
  };

  //***************************
  //Specialization for a vector
  //***************************
#define REF_FOR_VECTOR_ARGS std::vector<E>,typename refhelper::ValueTrait<std::vector<E> >::value,typename refhelper::FindTrait<std::vector<E>, typename refhelper::ValueTrait<std::vector<E> >::value>::value

  template <typename E>
  class Ref<REF_FOR_VECTOR_ARGS> {
  private:
    typedef typename refhelper::ValueTrait<std::vector<E> >::value T;
    typedef typename refhelper::FindTrait<std::vector<E>, typename refhelper::ValueTrait<std::vector<E> >::value>::value F;

    typedef refhelper::FindRefVectorUsingAdvance<RefVector<std::vector<E>, T, F> > VF;
    typedef refhelper::FindRefVectorUsingAdvance<RefToBaseVector<T> > VBF;
    friend class RefVectorIterator<std::vector<E>, T, F>;
    friend class RefVector<std::vector<E>, T, F>;
    friend class RefVector<RefVector<std::vector<E>, T, F>, T, VF>;
    friend class RefVector<RefVector<std::vector<E>, T, F>, T, VBF>;

  public:
    /// for export
    typedef std::vector<E> product_type;
    typedef typename refhelper::ValueTrait<std::vector<E> >::value value_type; 
    typedef value_type const element_type; //used for generic programming
    typedef typename refhelper::FindTrait<std::vector<E>,
                                          typename refhelper::ValueTrait<std::vector<E> >::value>::value finder_type;
    typedef typename boost::binary_traits<F>::second_argument_type argument_type;
    typedef unsigned int key_type;   
    /// C is the type of the collection
    /// T is the type of a member the collection
    
    static key_type invalidKey() { return key_traits<key_type>::value; }

    /// Default constructor needed for reading from persistent store. Not for direct use.
    Ref() : product_() {}
    
    /// General purpose constructor from handle.
    Ref(Handle<product_type> const& handle, key_type itemKey, bool setNow=true);
    
    /// General purpose constructor from orphan handle.
    Ref(OrphanHandle<product_type> const& handle, key_type itemKey, bool setNow=true);
    
    /// Constructors for ref to object that is not in an event.
    //  An exception will be thrown if an attempt is made to persistify
    //  any object containing this Ref.  Also, in the future work will
    //  be done to throw an exception if an attempt is made to put any object
    //  containing this Ref into an event(or run or lumi).
    Ref(product_type const* product, key_type itemKey, bool setNow=true);
    
    /// Constructor from test handle.
    //  An exception will be thrown if an attempt is made to persistify
    //  any object containing this Ref.  Also, in the future work will
    Ref(TestHandle<product_type> const& handle, key_type itemKey, bool setNow=true);
    
    /// Constructor for those users who do not have a product handle,
    /// but have a pointer to a product getter (such as the EventPrincipal).
    /// prodGetter will ususally be a pointer to the event principal.
    Ref(ProductID const& productID, key_type itemKey, EDProductGetter const* prodGetter) :
    product_(productID, 0, mustBeNonZero(prodGetter, "Ref", productID), false,itemKey) {
    }
    
    /// Constructor for use in the various X::fillView(...) functions.
    //  It is an error (not diagnosable at compile- or run-time) to call
    //  this constructor with a pointer to a T unless the pointed-to T
    //  object is already in a collection of type C stored in the
    //  Event. The given ProductID must be the id of the collection in
    //  the Event.
    
    Ref(ProductID const& iProductID, T const* item, key_type itemKey, product_type const* /* iProduct */) :
    product_(iProductID, item, 0, false, itemKey)
    { }

    Ref(ProductID const& iProductID, T const* item, key_type itemKey) :
    product_(iProductID, item, 0, false, itemKey)
    { }

    Ref(ProductID const& iProductID, T const* item, key_type itemKey, bool transient) :
    product_(iProductID, item, 0, transient, itemKey)
    { }

    /// Constructor that creates an invalid ("null") Ref that is
    /// associated with a given product (denoted by that product's
    /// ProductID).
    
    explicit Ref(ProductID const& iId) :
    product_(iId, 0, 0, false,key_traits<key_type>::value)
    { }
    
    /// Constructor from RefProd<C> and key
    Ref(RefProd<product_type> const& refProd, key_type itemKey);
    
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
    bool isNonnull() const { return key()!=edm::key_traits<key_type>::value; }
    
    /// Checks for null
    bool operator!() const {return isNull();}
    
    /// Accessor for product ID.
    ProductID id() const {return product_.id();}
    
    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {return product_.productGetter();}
    
    /// Accessor for product key.
    key_type key() const {return product_.index();}
    
    // This one just for backward compatibility.  Will be removed soon.
    key_type index() const {return product_.index();}
    
    /// Returns true if container referenced by the Ref has been cached
    bool hasProductCache() const {return product_.productPtr() != 0;}
    
    /// Checks if collection is in memory or available
    /// in the Event. No type checking is done.
    bool isAvailable() const;

    /// Checks if this ref is transient (i.e. not persistable).
    bool isTransient() const {return product_.isTransient();}

    RefCore const& refCore() const {return product_.toRefCore();}

    //Used by ROOT storage
    CMS_CLASS_VERSION(11)
    //  private:
    // Constructor from member of RefVector
    Ref(RefCore const& iRefCore, key_type const& iKey) : 
    product_(iRefCore,iKey) {
    }
    
  private:
    // Compile time check that the argument is a C* or C const*
    // or derived from it.
    void checkTypeAtCompileTime(product_type const*) {}

    RefCoreWithIndex product_;
  };
}

#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefCoreGet.h"
#include "DataFormats/Common/interface/RefItemGet.h"

namespace edm {

  /// General purpose constructor from handle.
  template <typename C, typename T, typename F>
  inline
  Ref<C, T, F>::Ref(Handle<C> const& handle, key_type itemKey, bool) :
    product_(handle.id(), nullptr, nullptr, false), index_(itemKey) {
    if(itemKey == key_traits<key_type>::value) return;
    refitem::findRefItem<C, T, F, key_type>(product_, handle.product(), itemKey);
  }

  /// General purpose constructor from handle.
  template <typename E>
  inline
  Ref<REF_FOR_VECTOR_ARGS>::Ref(Handle<std::vector<E> > const& handle, key_type itemKey, bool) :
    product_(handle.id(), nullptr, nullptr, false, itemKey){
    if(itemKey == key_traits<key_type>::value) return;
    refitem::findRefItem<product_type, value_type, finder_type, key_type>(product_.toRefCore(),
                                                                          handle.product(),
                                                                          itemKey);
  }

  /// General purpose constructor from orphan handle.
  template <typename C, typename T, typename F>
  inline
  Ref<C, T, F>::Ref(OrphanHandle<C> const& handle, key_type itemKey, bool) :
    product_(handle.id(), nullptr, nullptr, false), index_(itemKey) {
    if(itemKey == key_traits<key_type>::value) return;
    refitem::findRefItem<C, T, F, key_type>(product_, handle.product(), itemKey);
  }

  /// General purpose constructor from orphan handle.
  template <typename E>
  inline
  Ref<REF_FOR_VECTOR_ARGS>::Ref(OrphanHandle<std::vector<E> > const& handle, key_type itemKey, bool) :
    product_(handle.id(), nullptr, nullptr, false, itemKey){
    if(itemKey == key_traits<key_type>::value) return;
    refitem::findRefItem<product_type, value_type, finder_type, key_type>(product_.toRefCore(),
                                                                          handle.product(),
                                                                          itemKey);
  }
  
  /// Constructor for refs to object that is not in an event.
  //  An exception will be thrown if an attempt is made to persistify
  //  any object containing this Ref.  Also, in the future work will
  //  be done to throw an exception if an attempt is made to put any object
  //  containing this Ref into an event(or run or lumi).
  //  Note:  It is legal for the referenced object to be put into the event
  //  and persistified.  It is this Ref itself that cannot be persistified.
  template <typename C, typename T, typename F>
  inline
  Ref<C, T, F>::Ref(C const* iProduct, key_type itemKey, bool) :
    product_(ProductID(), nullptr, nullptr, true), index_(iProduct != nullptr ? itemKey : key_traits<key_type>::value) {
    if(iProduct != nullptr) {
      refitem::findRefItem<C, T, F, key_type>(product_, iProduct, itemKey);
    }
  }

  template <typename E>
  inline
  Ref<REF_FOR_VECTOR_ARGS>::Ref(std::vector<E> const* iProduct, key_type itemKey, bool) :
    product_(ProductID(), nullptr, nullptr, true, iProduct != 0 ? itemKey : key_traits<key_type>::value) {
    if(iProduct != nullptr) {
      refitem::findRefItem<product_type, value_type, finder_type, key_type>(product_.toRefCore(), iProduct, itemKey);
    }
  }

  /// constructor from test handle.
  //  An exception will be thrown if an attempt is made to persistify any object containing this Ref.
  template <typename C, typename T, typename F>
  inline
  Ref<C, T, F>::Ref(TestHandle<C> const& handle, key_type itemKey, bool) :
    product_(handle.id(), nullptr, nullptr, true), index_(itemKey) {
    if(itemKey == key_traits<key_type>::value) return;
    refitem::findRefItem<C, T, F, key_type>(product_, handle.product(), itemKey);
  }

  template <typename E>
  inline
  Ref<REF_FOR_VECTOR_ARGS>::Ref(TestHandle<std::vector<E> > const& handle, key_type itemKey, bool) :
    product_(handle.id(), nullptr, nullptr, true, itemKey){
    if(itemKey == key_traits<key_type>::value) return;
    refitem::findRefItem<product_type, value_type, finder_type, key_type>(product_.toRefCore(),
                                                                          handle.product(),
                                                                          itemKey);
  }

  /// Constructor from RefProd<C> and key
  template <typename C, typename T, typename F>
  inline
  Ref<C, T, F>::Ref(RefProd<C> const& refProd, key_type itemKey) :
    product_(refProd.id(), nullptr, refProd.refCore().productGetter(), refProd.refCore().isTransient()),
    index_(itemKey) {

    if(refProd.refCore().productPtr() != nullptr && itemKey != key_traits<key_type>::value) {
      refitem::findRefItem<C, T, F, key_type>(product_,
                                              static_cast<product_type const*>(refProd.refCore().productPtr()),
                                              itemKey);
    }
  }

  template <typename E>
  inline
  Ref<REF_FOR_VECTOR_ARGS>::Ref(RefProd<std::vector<E> > const& refProd, key_type itemKey) :
    product_(refProd.id(), nullptr, refProd.refCore().productGetter(), refProd.refCore().isTransient(), itemKey) {

    if(refProd.refCore().productPtr() != nullptr && itemKey != key_traits<key_type>::value) {
      refitem::findRefItem<product_type, value_type, finder_type, key_type>(
          product_.toRefCore(),
          static_cast<product_type const*>(refProd.refCore().productPtr()),
          itemKey);
    }
  }

  template <typename C, typename T, typename F>
  inline
  bool
  Ref<C, T, F>::isAvailable() const {
    if(product_.isAvailable()) {
      return true;
    }
    return isThinnedAvailable<C>(product_, index_);
  }

  template <typename E>
  inline
  bool
  Ref<REF_FOR_VECTOR_ARGS>::isAvailable() const {
    if(product_.isAvailable()) {
      return true;
    }
    return isThinnedAvailable<std::vector<E> >(product_.toRefCore(), key());
  }

  /// Dereference operator
  template <typename C, typename T, typename F>
  inline
  T const&
  Ref<C, T, F>::operator*() const {
    return *getRefPtr<C, T, F>(product_, index_);
  }
  template <typename E>
  inline
  typename refhelper::ValueTrait<std::vector<E> >::value const&
  Ref<REF_FOR_VECTOR_ARGS>::operator*() const {
    return *getRefPtr<REF_FOR_VECTOR_ARGS>(product_.toRefCore(), key());
  }

  /// Member dereference operator
  template <typename C, typename T, typename F>
  inline
  T const*
  Ref<C, T, F>::operator->() const {
    return getRefPtr<C, T, F>(product_, index_);
  }
  template <typename E>
  inline
  typename refhelper::ValueTrait<std::vector<E> >::value const*
  Ref<REF_FOR_VECTOR_ARGS>::operator->() const {
    return getRefPtr<REF_FOR_VECTOR_ARGS>(product_.toRefCore(), key());
  }

  template <typename C, typename T, typename F>
  inline
  bool
  operator==(Ref<C, T, F> const& lhs, Ref<C, T, F> const& rhs) {
    return lhs.key() == rhs.key() && lhs.refCore() == rhs.refCore() ;
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
    /// the definition and use of compare_key<> guarantees that the ordering of Refs within
      /// a collection will be identical to the ordering of the referenced objects in the collection.
      return (lhs.refCore() == rhs.refCore() ? compare_key<C>(lhs.key(), rhs.key()) : lhs.refCore() < rhs.refCore());
  }

}

//Handle specialization here
#include "DataFormats/Common/interface/HolderToVectorTrait_Ref_specialization.h"
#endif
