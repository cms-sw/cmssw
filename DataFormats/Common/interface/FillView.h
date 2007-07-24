#ifndef Common_FillView_h
#define Common_FillView_h

/*----------------------------------------------------------------------
  
Several fillView function templates, to provide View support for 
Standard Library containers.

$Id: FillView.h,v 1.3 2007/07/09 07:28:49 llista Exp $

----------------------------------------------------------------------*/

#include <string>
#include <vector>
#include <list>
#include <deque>
#include <set>


namespace edm {
  template<typename C, typename T, typename F> class RefVector;
  template<typename C, typename T, typename F> class Ref;
  namespace detail {

    // Function template reallyFillView<C> can fill views for
    // standard-library collections of appropriate types (vector, list,
    // deque, set).

    template<typename C>
    struct FillViewRefTypeTrait {
      typedef Ref<C> type;
    };
    
    template<typename C, typename T, typename F>
    struct FillViewRefTypeTrait<RefVector<C, T, F> > {
      typedef typename RefVector<C, T, F>::value_type type;
    };
  }
}

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {
  namespace detail {
    template <class COLLECTION>
    void
    reallyFillView(COLLECTION const& coll,
		   ProductID const& id,
		   std::vector<void const*>& ptrs,
		   helper_vector& helpers)
    {
      typedef COLLECTION                            product_type;
      typedef typename product_type::value_type     element_type;
      typedef typename product_type::const_iterator iter;
      typedef typename product_type::size_type      size_type;
      typedef typename FillViewRefTypeTrait<product_type>::type ref_type;
      typedef reftobase::RefHolder<ref_type>        holder_type;
      
      size_type key = 0;
      for (iter i = coll.begin(), e = coll.end(); i!=e; ++i, ++key) {
	element_type const* address = &*i;
	ptrs.push_back(address);
	holder_type h(ref_type(id, address, key));
	helpers.push_back(&h);
      }
    }
  }

  template <class T, class A>
  void
  fillView(std::vector<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   helper_vector& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

  template <class T, class A>
  void
  fillView(std::list<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   helper_vector& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

  template <class T, class A>
  void
  fillView(std::deque<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   helper_vector& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

  template <class T, class A, class Comp>
  void
  fillView(std::set<T,A,Comp> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   helper_vector& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

}

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBase.h"

#endif
