#ifndef Framework_View_h
#define Framework_View_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     View
// 
/**\class View View.h FWCore/Framework/interface/View.h

Description: Provide access to any EDProduct that is a sequence.


*/
//
// Original Author:  
//         Created:  Mon Dec 18 09:48:30 CST 2006
// $Id: View.h,v 1.4 2007/01/11 23:39:19 paterno Exp $
//

#include <algorithm>
#include <vector>

#include "boost/iterator/indirect_iterator.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/EDProduct.h"

namespace edm
{

  // ViewBase exists only so that we may invoke View<T> destructors
  // polymorphically.

  class ViewBase
  {
  public:
    virtual ~ViewBase();
  };

  template <class T>
  class View : public ViewBase
  {
    typedef std::vector<T const*>  seq_t;
  public:
    typedef T const*   pointer;
    typedef T const*   const_pointer;

    typedef T const&   reference;
    typedef T const&   const_reference;

    typedef T          value_type;

    typedef boost::indirect_iterator<typename seq_t::const_iterator> const_iterator;

    typedef typename seq_t::size_type size_type;
    typedef typename seq_t::difference_type difference_type;

    typedef boost::indirect_iterator<typename seq_t::const_reverse_iterator>  const_reverse_iterator;

    // Compiler-generated copy, and assignment each does the right
    // thing.

    View() : items_() { }

    // This function is dangerous, and should only be called from the
    // infrastructure code.
    explicit View(std::vector<void const*> const& pointers,
		  std::vector<helper_ptr> const& helpers) : 
      items_(),
      refs_()
    {
      size_type numElements = pointers.size();

      // If the two input vectors are not of the same size, there is a
      // logic error in the framework code that called this
      // constructor.
      assert (numElements == helpers.size());

      items_.reserve(numElements);
      refs_.reserve(numElements);
      for (std::vector<void const*>::size_type i = 0; i < pointers.size(); ++i)
	{
	  items_.push_back(static_cast<pointer>(pointers[i]));
	  refs_.push_back(RefToBase<T>(helpers[i]));
	}
      // Sanity check...
      assert(items_.size() == refs_.size());
    }

    virtual ~View() { }

    size_type capacity() const { return items_.capacity(); }

    // Most non-const member functions not present.
    // No access to non-const contents provided.

    const_iterator begin() const { return items_.begin(); }
    const_iterator end() const { return items_.end(); }

    const_reverse_iterator rbegin() const { return items_.rbegin(); }
    const_reverse_iterator rend() const { return items_.rend(); }

    size_type size() const { return items_.size(); }

    size_type max_size() const { return items_.max_size(); }

    bool empty() const { return items_.empty(); }

    const_reference at(size_type pos) const { return *items_.at(pos); }

    const_reference operator[](size_type pos) const { return *items_[pos]; }

    RefToBase<value_type> refAt(size_type i) const { return refs_[i]; }

    const_reference front() const { return *items_.front(); }

    const_reference back() const {return *items_.back(); }

    void pop_back() { items_.pop_back(); }

    // No erase, because erase is required to return an *iterator*,
    // not a *const_iterator*.

    // The following is for testing only.
    static void fill_from_range(T* first, T* last, View& output)
    {
      output.items_.resize(std::distance(first,last));
      for (typename View<T>::size_type i = 0; first != last; ++i, ++first)
	output.items_[i] = first;
    }

  private:
    seq_t                      items_;
    std::vector<RefToBase<T> > refs_;


  };

  // Associated free functions (same as for std::vector)
  template <class T> bool operator==(View<T> const&, View<T> const&);
  template <class T> bool operator!=(View<T> const&, View<T> const&);
  template <class T> bool operator< (View<T> const&, View<T> const&);
  template <class T> bool operator<=(View<T> const&, View<T> const&);
  template <class T> bool operator> (View<T> const&, View<T> const&);
  template <class T> bool operator>=(View<T> const&, View<T> const&);


  template <class T>
  inline
  bool
  operator== (View<T> const& lhs, View<T> const& rhs)
  {
    return 
      lhs.size() == rhs.size() &&
      std::equal(lhs.begin(), lhs.end(), rhs.begin());
  }

  template <class T>
  inline
  bool 
  operator!=(View<T> const& lhs, View<T> const& rhs)
  {
    return !(lhs==rhs);
  }

  template <class T>
  inline 
  bool 
  operator< (View<T> const& lhs, View<T> const& rhs)
  {
    return 
      std::lexicographical_compare(lhs.begin(), lhs.end(),
				   rhs.begin(), rhs.end());
  }

  template <class T> 
  inline
  bool
  operator<=(View<T> const& lhs, View<T> const& rhs)
  {
    return !(rhs<lhs);
  }

  template <class T> 
  inline 
  bool operator> (View<T> const& lhs, View<T> const& rhs)
  {
    return rhs<lhs;
  }

  template <class T>
  inline
  bool operator>=(View<T> const& lhs, View<T> const& rhs)
  {
    return !(lhs<rhs);
  }

}

#endif
