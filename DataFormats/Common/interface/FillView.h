#ifndef Common_FillView_h
#define Common_FillView_h

/*----------------------------------------------------------------------
  
Several fillView function templates, to provide View support for 
Standard Library containers.

$Id: Wrapper.h,v 1.16 2007/05/10 23:45:11 chrjones Exp $

----------------------------------------------------------------------*/

#include <string>
#include <vector>
#include <list>
#include <deque>
#include <set>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace edm {

  template <class T, class A>
  void
  fillView(std::vector<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   std::vector<helper_ptr>& helpers)
  {
    typedef std::vector<T,A>   product_type;
    typedef Ref<product_type>  ref_type;
    typedef typename product_type::const_iterator  iter;

    for (iter i = obj.begin(), e = obj.end(); i != e; ++i) 
      {
	ptrs.push_back(&*i);
	// How do we make the helper_ptr objects which point to the
	// elements of the vector?
	typedef reftobase::IndirectHolderHelper<ref_type> ihh_type;
	//helper_ptr p(new ihh_type(ref_type()));
	//helpers.push_back(p);
      }
    //throw edm::Exception(errors::UnimplementedFeature, "fillView(vector,...)");
  }

  template <class T, class A>
  void
  fillView(std::list<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   std::vector<helper_ptr>& helpers)
  {
    typedef typename std::list<T,A>::const_iterator iter;
    for (iter i = obj.begin(), e = obj.end(); i != e; ++i) ptrs.push_back(&*i);
    //throw edm::Exception(errors::UnimplementedFeature, "fillView(list,...)");
  }

  template <class T, class A>
  void
  fillView(std::deque<T,A> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   std::vector<helper_ptr>& helpers)
  {
    typedef typename std::deque<T,A>::const_iterator iter;
    for (iter i = obj.begin(), e = obj.end(); i != e; ++i) ptrs.push_back(&*i);
    //throw edm::Exception(errors::UnimplementedFeature, "fillView(deque,...)");
  }

  template <class T, class A, class Comp>
  void
  fillView(std::set<T,A,Comp> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& ptrs,
	   std::vector<helper_ptr>& helpers)
  {
    typedef typename std::set<T,A,Comp>::const_iterator iter;
    for (iter i = obj.begin(), e = obj.end(); i != e; ++i) ptrs.push_back(&*i);
    //throw edm::Exception(errors::UnimplementedFeature, "fillView(set,...)");
  }

}

#endif
