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
// $Id: View.h,v 1.2 2006/12/20 23:19:05 paterno Exp $
//

#include <algorithm>
#include <vector>

#include "boost/iterator/indirect_iterator.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm
{
  template <class T>
  class View
  {
    typedef std::vector<T*>  seq_t;
  public:
    typedef T*         pointer;
    typedef T const*   const_pointer;

    typedef T&         reference;
    typedef T const&   const_reference;

    typedef T          value_type;

    typedef boost::indirect_iterator<typename seq_t::const_iterator> const_iterator;

    typedef typename seq_t::size_type size_type;
    typedef typename seq_t::difference_type difference_type;

    typedef boost::indirect_iterator<typename seq_t::const_reverse_iterator>  const_reverse_iterator;

    // Compiler-generated default c'tor, copy, assignment, and d'tor
    // each does the right thing.

    size_type capacity() const { return data_.capacity(); }

    // Most non-const member functions not present.
    // No access to non-const contents provided.

    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

    const_reverse_iterator rbegin() const { return data_.rbegin(); }
    const_reverse_iterator rend() const { return data_.rend(); }

    size_type size() const { return data_.size(); }

    size_type max_size() const { return data_.max_size(); }

    bool empty() const { return data_.empty(); }

    const_reference at(size_type pos) const { return *data_.at(pos); }

    const_reference operator[](size_type pos) const { return *data_[pos]; }

    const_reference front() const { return *data_.front(); }

    const_reference back() const {return *data_.back(); }

    void pop_back() { data_.pop_back(); }

    // No erase, because erase is required to return an *iterator*,
    // not a *const_iterator*.

    // The following are for testing.
    static void fill_from_range(T* first, T* last, View& output)
    {
      output.data_.resize(std::distance(first,last));
      for (typename View<T>::size_type i = 0; first != last; ++i, ++first)
	output.data_[i] = first;
    }

  private:

    seq_t  data_;

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
