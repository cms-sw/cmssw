#ifndef Common_Wrapper_h
#define Common_Wrapper_h

/*----------------------------------------------------------------------
  
Wrapper: A template wrapper around EDProducts to hold the product ID.

$Id: Wrapper.h,v 1.19 2007/07/09 07:28:50 llista Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>
#include <list>
#include <deque>
#include <set>

#include "boost/mpl/if.hpp"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "DataFormats/Common/interface/traits.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

  template <class T>
  class Wrapper : public EDProduct {
  public:
    typedef T value_type;
    typedef T wrapped_type;  // used with Reflex to identify Wrappers
    Wrapper() : EDProduct(), present(false), obj() {}
    explicit Wrapper(std::auto_ptr<T> ptr);
    virtual ~Wrapper() {}
    T const * product() const {return (present ? &obj : 0);}
    T const * operator->() const {return product();}
    
    //these are used by FWLite
    static const std::type_info& productTypeInfo() { return typeid(T);}
    static const std::type_info& typeInfo() { return typeid(Wrapper<T>);}
    
  private:
    virtual bool isPresent_() const {return present;}
    virtual void do_fillView(ProductID const& id,
			     std::vector<void const*>& pointers,
			     helper_vector_ptr & helpers) const;
    // We wish to disallow copy construction and assignment.
    // We make the copy constructor and assignment operator private.
    Wrapper(Wrapper<T> const& rh); // disallow copy construction
    Wrapper<T> & operator=(Wrapper<T> const&); // disallow assignment
    bool present;
    //   T const obj;
    T obj;
  };

} //namespace edm

#include "DataFormats/Common/interface/Ref.h"

namespace edm {

  template <class T>
  struct DoFillView
  {
    void operator()(T const& obj,
		    ProductID const& id,
		    std::vector<void const*>& pointers,
		    helper_vector_ptr & helpers) const;
  };

  template <class T>
  struct DoNotFillView
  {
    void operator()(T const&,
		    ProductID const&,
		    std::vector<void const*>&,
		    helper_vector_ptr& ) const 
    {
      throw Exception(errors::ProductDoesNotSupportViews)
	<< "The product type " 
	<< typeid(T).name()
	<< "\ndoes not support Views\n";
    }
  };

    template <typename T>
    inline
    void Wrapper<T>::do_fillView(ProductID const& id,
			     std::vector<void const*>& pointers,
			     helper_vector_ptr& helpers) const
    {
      typename boost::mpl::if_c<has_fillView<T>::value,
	DoFillView<T>,
	DoNotFillView<T> >::type maybe_filler;
      maybe_filler(obj, id, pointers, helpers);
    }

  // This is an attempt to optimize for speed, by avoiding the copying
  // of large objects of type T. In this initial version, we assume
  // that for any class having a 'swap' member function should call
  // 'swap' rather than copying the object.

  template <typename T>
  struct DoSwap
  {
    void operator()(T& a, T& b) { a.swap(b); }
  };

  template <typename T>
  struct DoAssign
  {
    void operator()(T& a, T& b) { a = b; }
  };

  //------------------------------------------------------------
  // Metafunction support for compile-time selection of code used in
  // Wrapper constructor
  //

  namespace detail 
  {

#if GCC_PREREQUISITE(3,4,4)
    //------------------------------------------------------------
    // WHEN WE MOVE to a newer compiler version, the following code
    // should be activated. This code causes compilation failures under
    // GCC 3.2.3, because of a compiler error in dealing with our
    // application of SFINAE. GCC 3.4.2 is known to deal with this code
    // correctly.
    //------------------------------------------------------------
    typedef char (& no_tag)[1]; // type indicating FALSE
    typedef char (& yes_tag)[2]; // type indicating TRUE

    // Definitions for the following struct and function templates are
    // not needed; we only require the declarations.
    template <typename T, void (T::*)(T&)>  struct swap_function;
    template <typename T> no_tag  has_swap_helper(...);
    template <typename T> yes_tag has_swap_helper(swap_function<T, &T::swap> * dummy);

    template<typename T>
    struct has_swap_function
    {
      static bool const value = 
	sizeof(has_swap_helper<T>(0)) == sizeof(yes_tag);
    };
#else
    //------------------------------------------------------------
    // THE FOLLOWING SHOULD BE REMOVED when we move to a newer
    // compiler; see the note above.
    //------------------------------------------------------------
    // has_swap_function is a metafunction of one argument, the type T.
    // As with many metafunctions, it is implemented as a class with a data
    // member 'value', which contains the value 'returned' by the
    // metafunction.
    //
    // has_swap_function<T>::value is 'true' if T has the has_swap
    // member function (with the right signature), and 'false' if T has
    // no such member function.


    template<typename T>
    struct has_swap_function
    {
      static bool const value = has_swap<T>::value;	
    };
#endif
  }

  template <class T>
  Wrapper<T>::Wrapper(std::auto_ptr<T> ptr) :
    EDProduct(), 
    present(ptr.get() != 0),
    obj()
  { 
    if (present) {
      // The following will call swap if T has such a function,
      // and use assignment if T has no such function.
      typename boost::mpl::if_c<detail::has_swap_function<T>::value, 
	DoSwap<T>, 
	DoAssign<T> >::type swap_or_assign;
      swap_or_assign(obj, *ptr);	
    }
  }

}

#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

namespace edm {
  namespace helpers {
    template<typename T>
    struct ViewFiller {
      static void fill(T const& obj,
		       ProductID const& id,
		       std::vector<void const*>& pointers,
		       helper_vector_ptr & helpers) {
	/// the following shoudl work also if T is a RefVector<C>
	typedef Ref<T> ref;
	typedef RefVector<T, typename ref::value_type, typename ref::finder_type> ref_vector;
	helpers = helper_vector_ptr( new reftobase::RefVectorHolder<ref_vector> );
	// fillView is the name of an overload set; each concrete
	// collection T should supply a fillView function, in the same
	// namespace at that in which T is defined, or in the 'edm'
	// namespace.
	fillView(obj, id, pointers, * helpers);
	assert( pointers.size() == helpers->size());
     }
    };

    template<typename T>
    struct ViewFiller<RefToBaseVector<T> > {
      static void fill(RefToBaseVector<T> const& obj,
		       ProductID const& id,
		       std::vector<void const*>& pointers,
		       helper_vector_ptr & helpers) {
	std::auto_ptr<helper_vector> h = obj.vectorHolder();
	pointers.reserve( h->size() );
	// NOTE: the following implementation has unusual signature!
	fillView( obj, pointers );
	helpers = helper_vector_ptr( h );
      }
    };
  }

  template <class T>
  void DoFillView<T>::operator()(T const& obj,
				 ProductID const& id,
				 std::vector<void const*>& pointers,
				 helper_vector_ptr & helpers) const  {
    helpers::ViewFiller<T>::fill( obj, id, pointers, helpers );
  }

}

#include "DataFormats/Common/interface/FillView.h"

#endif
