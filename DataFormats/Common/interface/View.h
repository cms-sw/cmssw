#ifndef DataFormats_Common_View_h
#define DataFormats_Common_View_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     View
//
/**\class edm::View<T>

Description: Provide access to the collected elements contained by any EDProduct that is a sequence.

*/
//
// Original Author:
//         Created:  Mon Dec 18 09:48:30 CST 2006
//

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefVectorHolderBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

#include "boost/iterator/indirect_iterator.hpp"

#include <vector>

namespace edm {

  //------------------------------------------------------------------
  // Class ViewBase
  //
  // ViewBase is an abstract base class. It exists only so that we
  // make invoke View<T> destructors polymorphically, and copy them
  // using clone().
  //
  //------------------------------------------------------------------

  class ViewBase {
  public:
    virtual ~ViewBase();
    ViewBase* clone() const;

  protected:
    ViewBase();
    ViewBase(ViewBase const&);
    virtual ViewBase* doClone() const = 0;
    void swap(ViewBase& other) {} // Nothing to swap
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


  template<typename T>
  class View : public ViewBase {
    typedef std::vector<T const*>  seq_t;
  public:
    typedef T const*   pointer;
    typedef T const*   const_pointer;

    typedef T const&   reference;
    typedef T const&   const_reference;

    typedef T          value_type;

    typedef boost::indirect_iterator<typename seq_t::const_iterator> const_iterator;

    // This should be a typedef to seq_t::size_type but because this type is used as a template
    // argument in a persistened class it must be stable for different architectures
    typedef unsigned int  size_type;
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

    View& operator=(View const& rhs);

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
    Ptr<value_type> ptrAt(size_type i) const;
    RefToBaseVector<T> const& refVector() const { return refs_; }
    PtrVector<T> const& ptrVector() const { return ptrs_; }

    const_reference front() const;
    const_reference back() const;
    void pop_back();
    ProductID id() const;
    EDProductGetter const* productGetter() const;

    // No erase, because erase is required to return an *iterator*,
    // not a *const_iterator*.

    // The following is for testing only.
    static void fill_from_range(T* first, T* last, View& output);

    void const* product() const {
      return refs_.product();
    }

  private:
    seq_t items_;
    RefToBaseVector<T> refs_;
    PtrVector<T> ptrs_;
    ViewBase* doClone() const;
  };

  // Associated free functions (same as for std::vector)
  template<typename T> bool operator==(View<T> const&, View<T> const&);
  template<typename T> bool operator!=(View<T> const&, View<T> const&);
  template<typename T> bool operator< (View<T> const&, View<T> const&);
  template<typename T> bool operator<=(View<T> const&, View<T> const&);
  template<typename T> bool operator> (View<T> const&, View<T> const&);
  template<typename T> bool operator>=(View<T> const&, View<T> const&);

  //------------------------------------------------------------------
  // Implementation of View<T>
  //------------------------------------------------------------------

  template<typename T>
  inline
  View<T>::View() :
    items_(),
    refs_(),
    ptrs_() {
  }

  template<typename T>
  View<T>::View(std::vector<void const*> const& pointers,
                helper_vector_ptr const& helpers) :
    items_(),
    refs_(),
    ptrs_() {
    size_type numElements = pointers.size();

    // If the two input vectors are not of the same size, there is a
    // logic error in the framework code that called this.
    // constructor.
    if(helpers.get() != 0) {
      assert(numElements == helpers->size());

      items_.reserve(numElements);
       ptrs_.reserve(refs_.size());
      for(std::vector<void const*>::size_type i = 0; i < pointers.size(); ++i) {
        void const* p = pointers[i];
        items_.push_back(static_cast<pointer>(p));
        if(0!=p) {
           ptrs_.push_back(Ptr<T>(helpers->id(), static_cast<T const*>(p), helpers->keyForIndex(i)));
        } else if(helpers->productGetter() != 0) {
           ptrs_.push_back(Ptr<T>(helpers->id(), helpers->keyForIndex(i), helpers->productGetter()));
        } else {
           ptrs_.push_back(Ptr<T>(helpers->id(), 0, helpers->keyForIndex(i)));
        }
      }
      RefToBaseVector<T> temp(helpers);
      refs_.swap(temp);
    }
  }

  template<typename T>
  View<T>::~View() {
  }

  template<typename T>
  inline
  void
  View<T>::swap(View& other) {
    this->ViewBase::swap(other);
    items_.swap(other.items_);
    refs_.swap(other.refs_);
    ptrs_.swap(other.ptrs_);
  }

  template<typename T>
  inline
  typename  View<T>::size_type
  View<T>::capacity() const {
    return items_.capacity();
  }

  template<typename T>
  inline
  typename View<T>::const_iterator
  View<T>::begin() const {
    return items_.begin();
  }

  template<typename T>
  inline
  typename View<T>::const_iterator
  View<T>::end() const {
    return items_.end();
  }

  template<typename T>
  inline
  typename View<T>::const_reverse_iterator
  View<T>::rbegin() const {
    return items_.rbegin();
  }

  template<typename T>
  inline
  typename View<T>::const_reverse_iterator
  View<T>::rend() const {
    return items_.rend();
  }

  template<typename T>
  inline
  typename View<T>::size_type
  View<T>::size() const {
    return items_.size();
  }

  template<typename T>
  inline
  typename View<T>::size_type
  View<T>::max_size() const {
    return items_.max_size();
  }

  template<typename T>
  inline
  bool
  View<T>::empty() const {
    return items_.empty();
  }

  template<typename T>
  inline
  typename View<T>::const_reference
  View<T>::at(size_type pos) const {
    return *items_.at(pos);
  }

  template<typename T>
  inline
  typename View<T>::const_reference
  View<T>::operator[](size_type pos) const {
    return *items_[pos];
  }

  template<typename T>
  inline
  RefToBase<T>
  View<T>::refAt(size_type i) const {
    return refs_[i];
  }

  template<typename T>
  inline
  Ptr<T>
  View<T>::ptrAt(size_type i) const {
    RefToBase<T> ref = refAt(i);
    return Ptr<T>(ref.id(), (ref.isAvailable() ? ref.get(): 0), ref.key());
  }

  template<typename T>
  inline
  typename View<T>::const_reference
  View<T>::front() const {
    return *items_.front();
  }

  template<typename T>
  inline
  typename View<T>::const_reference
  View<T>::back() const {
    return *items_.back();
  }

  template<typename T>
  inline
  void
  View<T>::pop_back() {
    items_.pop_back();
  }

  template<typename T>
  inline
  ProductID
  View<T>::id() const {
    return refs_.id();
  }
  template<typename T>
  inline
  EDProductGetter const*
  View<T>::productGetter() const {
    return refs_.productGetter();
  }

  // The following is for testing only.
  template<typename T>
  inline
  void
  View<T>::fill_from_range(T* first, T* last, View& output) {
    output.items_.resize(std::distance(first, last));
    for(typename View<T>::size_type i = 0; first != last; ++i, ++first)
      output.items_[i] = first;
  }

  template<typename T>
  ViewBase*
  View<T>::doClone() const {
    return new View(*this);
  }

  template<typename T>
  inline
  View<T>&
  View<T>::operator=(View<T> const& rhs) {
    View<T> temp(rhs);
    this->swap(temp);
    return *this;
  }

  template<typename T>
  inline
  bool
  operator==(View<T> const& lhs, View<T> const& rhs) {
    return
      lhs.size() == rhs.size() &&
      std::equal(lhs.begin(), lhs.end(), rhs.begin());
  }

  template<typename T>
  inline
  bool
  operator!=(View<T> const& lhs, View<T> const& rhs) {
    return !(lhs == rhs);
  }

  template<typename T>
  inline
  bool
  operator<(View<T> const& lhs, View<T> const& rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                        rhs.begin(), rhs.end());
  }

  template<typename T>
  inline
  bool
  operator<=(View<T> const& lhs, View<T> const& rhs) {
    return !(rhs<lhs);
  }

  template<typename T>
  inline
  bool operator> (View<T> const& lhs, View<T> const& rhs) {
    return rhs<lhs;
  }

  template<typename T>
  inline
  bool operator>=(View<T> const& lhs, View<T> const& rhs) {
    return !(lhs<rhs);
  }

  // Free swap function
  template<typename T>
  inline
  void swap(View<T>& lhs, View<T>& rhs) {
    lhs.swap(rhs);
  }
}

#endif
