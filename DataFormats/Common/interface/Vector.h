#ifndef DataFormats_Common_Vector_h
#define DataFormats_Common_Vector_h
// $Id: Vector.h,v 1.3 2008/03/31 21:12:11 wmtan Exp $

#include <algorithm>
#include <functional>
#include <vector>

#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/Ref.h"

#if defined CMS_USE_DEBUGGING_ALLOCATOR
#include "DataFormats/Common/interface/debugging_allocator.h"
#endif

#include "DataFormats/Common/interface/PostReadFixupTrait.h"

namespace edm {
  class ProductID;
  template <typename T>
  class Vector  {
  private:
#if defined(CMS_USE_DEBUGGING_ALLOCATOR)
    typedef std::vector<T, debugging_allocator<T> > base;
#else
    typedef std::vector<T> base;
#endif

  public:
    typedef typename base::size_type size_type;
    typedef typename base::value_type value_type;
    typedef typename base::pointer pointer;
    typedef typename base::reference reference;
    typedef typename base::const_reference const_reference;
      
    typedef typename base::iterator iterator;
    typedef typename base::const_iterator const_iterator;

    Vector();
    Vector(size_type);
    Vector(const Vector &);
    ~Vector();
      
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    size_type size() const;
    bool empty() const;
    reference operator[](size_type);
    const_reference operator[](size_type) const;
    Vector<T> & operator=(const Vector<T> &);
    void reserve(size_t);
    void push_back(const value_type &);
    void pop_back();
    reference back();
    const_reference back() const;
    reference front();
    const_reference front() const;
    const base & data() const; 
    void clear();
    iterator erase(iterator pos);
    iterator erase(iterator first, iterator last);
    void swap(Vector<T> & other);

    void fillView(ProductID const& id,
		  std::vector<void const*>& pointers,
		  helper_vector& helpers) const;

  private:
    base data_;      
    typename helpers::PostReadFixupTrait<T>::type fixup_;
    inline void fixup() const { fixup_(data_); }
    inline void touch() { fixup_.touch(); }
  };
  
  template<typename T>
  inline Vector<T>::Vector() : data_() { 
  }
  
  template<typename T>
  inline Vector<T>::Vector(size_type n) : data_(n) { 
  }
  
  template<typename T>
  inline Vector<T>::Vector(const Vector<T> & o) : data_(o.data_) {
  }
  
  template<typename T>
  inline Vector<T>::~Vector() { 
  }
  
  template<typename T>
  inline Vector<T> & Vector<T>::operator=(const Vector<T> & o) {
    Vector<T> temp(o);
    this->swap(temp);
    return *this;
  }
  template<typename T>
  inline typename Vector<T>::iterator Vector<T>::begin() {
    fixup();
    touch();
    return iterator(data_.begin());
  }
  
  template<typename T>
  inline typename Vector<T>::iterator Vector<T>::end() {
    fixup();
    touch();
    return iterator(data_.end());
  }
  
  template<typename T>
  inline typename Vector<T>::const_iterator Vector<T>::begin() const {
    fixup();
    return const_iterator(data_.begin());
  }
  
  template<typename T>
  inline typename Vector<T>::const_iterator Vector<T>::end() const {
    fixup();
    return const_iterator(data_.end());
  }
  
  template<typename T>
  inline typename Vector<T>::size_type Vector<T>::size() const {
    return data_.size();
  }
  
  template<typename T>
  inline bool Vector<T>::empty() const {
    return data_.empty();
  }
  
  template<typename T>
  inline typename Vector<T>::reference Vector<T>::operator[](size_type n) {
    fixup();
    return data_[n];
  }
  
  template<typename T>
  inline typename Vector<T>::const_reference Vector<T>::operator[](size_type n) const {
    fixup();
    return data_[n];
  }
  
  template<typename T>
  inline void Vector<T>::reserve(size_t n) {
    data_.reserve(n);
  }
  
  template<typename T>
  inline void Vector<T>::push_back(const typename Vector<T>::value_type & d) {
    data_.push_back(d);
    touch();
  }

  template<typename T>
  inline void Vector<T>::pop_back() {
    // We have to delete the pointed-to thing, before we squeeze it
    // out of the vector...
    data_.pop_back();
    touch();
  }

  template<typename T>
  inline typename Vector<T>::reference Vector<T>::back() {
    fixup();
    touch();
    return data_.back();
  }
  
  template<typename T>
  inline typename Vector<T>::const_reference Vector<T>::back() const {
    fixup();
    return data_.back();
  }
  
  template<typename T>
  inline typename Vector<T>::reference Vector<T>::front() {
    fixup();
    touch();
    return data_.front();
  }
  
  template<typename T>
  inline typename Vector<T>::const_reference Vector<T>::front() const {
    fixup();
    return data_.front();
  }
  
  template<typename T>
  inline const typename Vector<T>::base & Vector<T>::data() const {
    fixup();
    return data_;
  }

  template<typename T>
  inline void Vector<T>::clear() {
    data_.clear();
  }

  template<typename T>
  typename Vector<T>::iterator Vector<T>::erase(iterator pos) {
    fixup();
    touch();
    return data_.erase(pos);
  }

  template<typename T>
  typename Vector<T>::iterator Vector<T>::erase(iterator first, iterator last) {
    fixup();
    touch();
    return data_.erase(first, last);
  }

  template<typename T>
  inline void Vector<T>::swap(Vector<T>& other) {
    data_.swap(other.data_);
    std::swap(fixup_, other.fixup_);
  }

  template<typename T>
  void Vector<T>::fillView(ProductID const& id,
			   std::vector<void const*>& pointers,
			   helper_vector& helpers) const
  {
    typedef Ref<Vector>      ref_type ;
    typedef reftobase::RefHolder<ref_type> holder_type;

    size_type numElements = this->size();
    pointers.reserve(numElements);
    helpers.reserve(numElements);
    size_type key = 0;
    for(typename base::const_iterator i=data_.begin(), e=data_.end(); i!=e; ++i, ++key) {

      pointers.push_back(&*i);
      holder_type h(ref_type(id, &*i, key));
      helpers.push_back(&h);
    }
  }

  template<typename T>
  inline void swap(Vector<T>& a, Vector<T>& b) {
    a.swap(b);
  }

  //----------------------------------------------------------------------
  //
  // Free function template to support creation of Views.

  template <typename T>
  inline
  void
  fillView(Vector<T> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& pointers,
	   helper_vector& helpers) {
    obj.fillView(id, pointers, helpers);
  }

  template <typename T>
  struct has_fillView<edm::Vector<T> >
  {
    static bool const value = true;
  };

}

#endif
