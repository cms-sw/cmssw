#ifndef DataFormats_Common_FillView_h
#define DataFormats_Common_FillView_h


/*----------------------------------------------------------------------
  
Several fillView function templates, to provide View support for 
Standard Library containers.

----------------------------------------------------------------------*/

#include <vector>
#include <list>
#include <deque>
#include <set>
#include "DataFormats/Common/interface/GetProduct.h"
#include "DataFormats/Common/interface/FillViewHelperVector.h"

namespace edm {

  class ProductID;

  namespace detail {
#ifndef __GCCXML__

    template <class COLLECTION>
    void
    reallyFillView(COLLECTION const& coll,
		   ProductID const& id,
		   std::vector<void const*>& ptrs,
		   FillViewHelperVector& helpers)
    {
      typedef COLLECTION                            product_type;
      typedef typename GetProduct<product_type>::element_type     element_type;
      typedef typename product_type::const_iterator iter;
      typedef typename product_type::size_type      size_type;
      
      ptrs.reserve(ptrs.size() + coll.size());
      helpers.reserve(ptrs.size() + coll.size());
      size_type key = 0;
      for (iter i = coll.begin(), e = coll.end(); i!=e; ++i, ++key) {
        element_type const* address = GetProduct<product_type>::address(i);
        ptrs.push_back(address);
        helpers.emplace_back(id,key);
      }
    }
#else
    template <class COLLECTION>
    void
    reallyFillView(COLLECTION const& coll,
                   ProductID const& id,
                   std::vector<void const*>& ptrs,
                   FillViewHelperVector& helpers);
#endif
  }
  template <class T, class A>
  void
  fillView(std::vector<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   FillViewHelperVector& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

  template <class T, class A>
  void
  fillView(std::list<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   FillViewHelperVector& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

  template <class T, class A>
  void
  fillView(std::deque<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   FillViewHelperVector& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

  template <class T, class A, class Comp>
  void
  fillView(std::set<T,A,Comp> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   FillViewHelperVector& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

}

#endif
