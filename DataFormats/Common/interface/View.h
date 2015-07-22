#ifndef DataFormats_Common_View_h
#define DataFormats_Common_View_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     View
//
/**\class edm::View<T>

Description: Provide access to the collected elements contained by any WrapperBase that is a sequence.

*/
//
// Original Author:
//         Created:  Mon Dec 18 09:48:30 CST 2006
//

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/IndirectHolder.h"
#include "DataFormats/Common/interface/RefHolder_.h"
#include "boost/iterator/indirect_iterator.hpp"

#include <vector>
#include <memory>

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
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    std::unique_ptr<ViewBase> clone() const;
#endif

  protected:
    ViewBase();
    ViewBase(ViewBase const&);
    ViewBase& operator=(ViewBase const&);
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    virtual std::unique_ptr<ViewBase> doClone() const = 0;
#endif
    void swap(ViewBase&) {} // Nothing to swap
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
         FillViewHelperVector const& helpers,
         EDProductGetter const* getter);

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
    std::vector<Ptr<value_type> > const& ptrs() const;

    const_reference front() const;
    const_reference back() const;

    // No erase, because erase is required to return an *iterator*,
    // not a *const_iterator*.

    // The following is for testing only.
    static void fill_from_range(T* first, T* last, View& output);

  private:
    seq_t items_;
    std::vector<Ptr<value_type> > vPtrs_;
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    std::unique_ptr<ViewBase> doClone() const override;
#endif
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
    vPtrs_() {
  }

#ifndef __GCCXML__
  template<typename T>
  View<T>::View(std::vector<void const*> const& pointers,
                FillViewHelperVector const& helpers,
                EDProductGetter const* getter) :
    items_(),
    vPtrs_() {
    size_type numElements = pointers.size();

    // If the two input vectors are not of the same size, there is a
    // logic error in the framework code that called this.
    // constructor.
    assert(numElements == helpers.size());

    items_.reserve(numElements);
    vPtrs_.reserve(numElements);
    for(std::vector<void const*>::size_type i = 0; i < pointers.size(); ++i) {
      void const* p = pointers[i];
      auto const& h = helpers[i];
      items_.push_back(static_cast<pointer>(p));
      if(0!=p) {
         vPtrs_.push_back(Ptr<T>(h.first, static_cast<T const*>(p), h.second));
      } else if(getter != nullptr) {
         vPtrs_.push_back(Ptr<T>(h.first, h.second, getter));
      } else {
         vPtrs_.push_back(Ptr<T>(h.first, nullptr, h.second));
      }
    }
  }
#endif

  template<typename T>
  View<T>::~View() {
  }

  template<typename T>
  inline
  void
  View<T>::swap(View& other) {
    this->ViewBase::swap(other);
    items_.swap(other.items_);
    vPtrs_.swap(other.vPtrs_);
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

#ifndef __GCCXML__
  template<typename T>
  inline
  RefToBase<T>
  View<T>::refAt(size_type i) const {
    //NOTE: considered creating a special BaseHolder for edm::Ptr.
    // But the IndirectHolder and RefHolder would still be needed
    // for other reasons. To reduce the number of dictionaries needed
    // we avoid using a more efficient BaseHolder.
    return RefToBase<T>(std::unique_ptr<reftobase::BaseHolder<T>>{
                          new reftobase::IndirectHolder<T>{
                            std::unique_ptr<reftobase::RefHolder<edm::Ptr<T>>>{
                              new reftobase::RefHolder<Ptr<T>>{ptrAt(i)}
                            }
                          }
                        } );
  }
#endif
  
  template<typename T>
  inline
  Ptr<T>
  View<T>::ptrAt(size_type i) const {
    return vPtrs_[i];
  }
  
  template<typename T>
  inline
  std::vector<Ptr<T> > const&
  View<T>::ptrs() const {
    return vPtrs_;
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

  // The following is for testing only.
  template<typename T>
  inline
  void
  View<T>::fill_from_range(T* first, T* last, View& output) {
    output.items_.resize(std::distance(first, last));
    for(typename View<T>::size_type i = 0; first != last; ++i, ++first)
      output.items_[i] = first;
  }

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  template<typename T>
  std::unique_ptr<ViewBase>
  View<T>::doClone() const {
    return std::unique_ptr<ViewBase>{new View(*this)};
  }
#endif
  
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
