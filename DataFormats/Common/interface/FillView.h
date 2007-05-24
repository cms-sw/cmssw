#ifndef Common_FillView_h
#define Common_FillView_h

/*----------------------------------------------------------------------
  
Several fillView function templates, to provide View support for 
Standard Library containers.

$Id: FillView.h,v 1.1 2007/05/16 22:31:59 paterno Exp $

----------------------------------------------------------------------*/

#include <string>
#include <vector>
#include <list>
#include <deque>
#include <set>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace edm 
{

  namespace detail 
  {

    // Function template reallyFillView<C> can fill views for
    // standard-library collections of appropriate types (vector, list,
    // deque, set).
    
    template <class COLLECTION>
    void
    reallyFillView(COLLECTION const& coll,
		   ProductID const& id,
		   std::vector<void const*>& ptrs,
		   std::vector<helper_ptr>& helpers)
    {
      typedef COLLECTION                            product_type;
      typedef typename product_type::value_type     element_type;
      typedef typename product_type::const_iterator iter;
      typedef typename product_type::size_type      size_type;
      typedef Ref<product_type>                     ref_type;
      typedef reftobase::RefHolder<ref_type>        holder_type;
      
      size_type key = 0;
      for (iter i = coll.begin(), e = coll.end(); i!=e; ++i, ++key)
	{
	  element_type const* address = &*i;
	  ptrs.push_back(address);
	  helper_ptr ptr(new holder_type(ref_type(id, address, key)));
	  helpers.push_back(ptr);
	}
    }
  }

  template <class T, class A>
  void
  fillView(std::vector<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   std::vector<helper_ptr>& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

  template <class T, class A>
  void
  fillView(std::list<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   std::vector<helper_ptr>& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

  template <class T, class A>
  void
  fillView(std::deque<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   std::vector<helper_ptr>& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

  template <class T, class A, class Comp>
  void
  fillView(std::set<T,A,Comp> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   std::vector<helper_ptr>& helpers)
  {
    detail::reallyFillView(obj, id, ptrs, helpers);
  }

}

#endif
