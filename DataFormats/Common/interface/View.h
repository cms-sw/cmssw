#ifndef Common_View_h
#define Common_View_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     View
// 
/**\class edm::View<T> 

Description: Provide access to the collected elements contained by any
EDProduct that is a sequence.

*/
//
// Original Author:  
//         Created:  Mon Dec 18 09:48:30 CST 2006
// $Id: View.h,v 1.8 2007/06/20 15:54:13 paterno Exp $
//

#include <vector>

#include "boost/iterator/indirect_iterator.hpp"

#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/RefVectorHolderBase.h"

namespace edm
{

  //------------------------------------------------------------------
  // Class ViewBase
  //
  // ViewBase is an abstract base class. It exists only so that we
  // make invoke View<T> destructors polymorphically, and copy them
  // using clone().
  // 
  //------------------------------------------------------------------

  class ViewBase
  {
  public:
    virtual ~ViewBase();
    ViewBase* clone() const;

  protected:
    ViewBase();
    ViewBase(ViewBase const&);
    virtual ViewBase* doClone() const = 0;    
  };

  //------------------------------------------------------------------
  /// Class template View<T>
  ///
  /// View<T> provides a way to allow reference to the elements (of
  /// type T) of some collection in an Event, without knowing about the
  /// type of the collection itself. For example, View<int> can refer
  /// to the ints in either a vector<int> or a list<int>, without the
  /// client code knowing about which type of container manages the
  /// ints.
  ///
  /// View<T> is not persistable.
  ///
  /// View<T> can be used to reference objects of any type that has T
  /// as a public base.
  ///
  //------------------------------------------------------------------


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

    typedef boost::indirect_iterator<typename seq_t::const_reverse_iterator> const_reverse_iterator;

    // Compiler-generated copy, and assignment each does the right
    // thing.

    View();

    // This function is dangerous, and should only be called from the
    // infrastructure code.
    View(std::vector<void const*> const& pointers,
	 helper_vector_ptr const& helpers);

    virtual ~View();

    void swap(View& other);

    size_type capacity() const;

    // Most non-const member functions not present.
    // No access to non-const contents provided.

    const_iterator begin() const;
    const_iterator end() const;

    const_reverse_iterator rbegin() const;
    const_reverse_iterator rend() const;

    size_type size() const;
    size_type max_size() const;
    bool empty() const;
    const_reference at(size_type pos) const;
    const_reference operator[](size_type pos) const;
    RefToBase<value_type> refAt(size_type i) const;
    const RefToBaseVector<T> & refVector() const { return refs_; }

    const_reference front() const;
    const_reference back() const;
    void pop_back();
    ProductID id() const;
    EDProductGetter const* productGetter() const;

    // No erase, because erase is required to return an *iterator*,
    // not a *const_iterator*.

    // The following is for testing only.
    static void fill_from_range(T* first, T* last, View& output);

  private:
    seq_t items_;
    RefToBaseVector<T> refs_;
    ViewBase* doClone() const;
  };

  // Associated free functions (same as for std::vector)
  template <class T> bool operator==(View<T> const&, View<T> const&);
  template <class T> bool operator!=(View<T> const&, View<T> const&);
  template <class T> bool operator< (View<T> const&, View<T> const&);
  template <class T> bool operator<=(View<T> const&, View<T> const&);
  template <class T> bool operator> (View<T> const&, View<T> const&);
  template <class T> bool operator>=(View<T> const&, View<T> const&);

  //------------------------------------------------------------------
  // Implementation of View<T>
  //------------------------------------------------------------------

  template <class T>
  inline
  View<T>::View() : 
    items_(),
    refs_()
  { }

  template <class T>
  View<T>::View(std::vector<void const*> const& pointers,
		helper_vector_ptr const& helpers) : 
    items_(),
    refs_()
  {
    size_type numElements = pointers.size();

    // If the two input vectors are not of the same size, there is a
    // logic error in the framework code that called this.
    // constructor.
    assert (numElements == helpers->size());

    items_.reserve(numElements);
    for (std::vector<void const*>::size_type i = 0; i < pointers.size(); ++i) {
	items_.push_back(static_cast<pointer>(pointers[i]));
    }
    refs_ = RefToBaseVector<T>( helpers );
  }

  template <class T>
  View<T>::~View() 
  { }

  template <class T>
  inline
  void
  View<T>::swap(View& other)
  {
    swap(items_, other.items_);
    swap(refs_, other.refs_);
  }

  template <class T>
  inline
  typename  View<T>::size_type 
  View<T>::capacity() const 
  {
    return items_.capacity();
  }

  template <class T>
  inline
  typename View<T>::const_iterator 
  View<T>::begin() const 
  {
    return items_.begin();
  }

  template <class T>
  inline
  typename View<T>::const_iterator 
  View<T>::end() const
  {
    return items_.end();
  }

  template <class T>
  inline
  typename View<T>::const_reverse_iterator 
  View<T>::rbegin() const
  {
    return items_.rbegin();
  }

  template <class T>
  inline
  typename View<T>::const_reverse_iterator
  View<T>::rend() const
  {
    return items_.rend();
  }

  template <class T>
  inline
  typename View<T>::size_type
  View<T>::size() const 
  {
    return items_.size();
  }

  template <class T>
  inline
  typename View<T>::size_type
  View<T>::max_size() const
  {
    return items_.max_size();
  }

  template <class T>
  inline
  bool 
  View<T>::empty() const 
  {
    return items_.empty();
  }

  template <class T>
  inline
  typename View<T>::const_reference 
  View<T>::at(size_type pos) const
  {
    return *items_.at(pos);
  }

  template <class T>
  inline
  typename View<T>::const_reference 
  View<T>::operator[](size_type pos) const
  {
    return *items_[pos];
  }

  template <class T>
  inline
  RefToBase<T> 
  View<T>::refAt(size_type i) const
  {
    return refs_[i];
  }

  template <class T>
  inline
  typename View<T>::const_reference 
  View<T>::front() const
  {
    return *items_.front();
  }

  template <class T>
  inline
  typename View<T>::const_reference
  View<T>::back() const
  {
    return *items_.back();
  }

  template <class T>
  inline
  void
  View<T>::pop_back()
  {
    items_.pop_back();
  }

  template <class T>
  inline 
  ProductID 
  View<T>::id() const {
    return refs_.id();
  }
  template <class T>
  inline
  EDProductGetter const* 
  View<T>::productGetter() const {
    return refs_.productGetter();
  }

  // The following is for testing only.
  template <class T>
  inline
  void
  View<T>::fill_from_range(T* first, T* last, View& output)
  {
    output.items_.resize(std::distance(first,last));
    for (typename View<T>::size_type i = 0; first != last; ++i, ++first)
      output.items_[i] = first;
  }

  template <class T>
  ViewBase*
  View<T>::doClone() const
  {
    return new View(*this);
  }

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
