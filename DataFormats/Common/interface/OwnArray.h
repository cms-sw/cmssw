#ifndef DataFormats_Common_OwnArray_h
#define DataFormats_Common_OwnArray_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/fillPtrVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/setPtr.h"
#include "DataFormats/Common/interface/traits.h"

#if defined CMS_USE_DEBUGGING_ALLOCATOR
#include "DataFormats/Common/interface/debugging_allocator.h"
#endif
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <functional>
#include <typeinfo>
#include <vector>

namespace edm {
  class ProductID;
  template <typename T, unsigned int MAX_SIZE, typename P = ClonePolicy<T> >
  class OwnArray {
  private:
    typedef OwnArray<T,MAX_SIZE,P> self;
    typedef std::vector<T*> base;
  public:
    typedef unsigned int size_type;
    typedef T value_type;
    typedef T* pointer;
    typedef T& reference;
    typedef T const& const_reference;
    typedef P policy_type;
    
    class iterator;
    class const_iterator {
    public:
      typedef T value_type;
      typedef T* pointer;
      typedef T const& reference;
      typedef ptrdiff_t difference_type;
      typedef typename base::const_iterator::iterator_category iterator_category;
      const_iterator(pointer const *  it) : i(it) { }
      const_iterator(iterator const& it) : i(it.i) { }
      const_iterator() {}
      const_iterator& operator++() { ++i; return *this; }
      const_iterator operator++(int) { const_iterator ci = *this; ++i; return ci; }
      const_iterator& operator--() { --i; return *this; }
      const_iterator operator--(int) { const_iterator ci = *this; --i; return ci; }
      difference_type operator-(const_iterator const& o) const { return i - o.i; }
      const_iterator operator+(difference_type n) const { return const_iterator(i + n); }
      const_iterator operator-(difference_type n) const { return const_iterator(i - n); }
      bool operator<(const_iterator const& o) const { return i < o.i; }
      bool operator==(const_iterator const& ci) const { return i == ci.i; }
      bool operator!=(const_iterator const& ci) const { return i != ci.i; }
      T const& operator *() const { return **i; }
      //    operator T const*() const { return & **i; }
      T const* operator->() const { return & (operator*()); }
      const_iterator & operator +=(difference_type d) { i += d; return *this; }
      const_iterator & operator -=(difference_type d) { i -= d; return *this; }
      reference operator[](difference_type d) const { return *const_iterator(i+d); } // for boost::iterator_range []
    private:
      pointer const * i;
    };
    class iterator {
    public:
      typedef T value_type;
      typedef T * pointer;
      typedef T & reference;
      typedef ptrdiff_t difference_type;
      typedef typename base::iterator::iterator_category iterator_category;
      iterator(pointer * it) : i(it) { }
      iterator() {}
      iterator& operator++() { ++i; return *this; }
      iterator operator++(int) { iterator ci = *this; ++i; return ci; }
      iterator& operator--() { --i; return *this; }
      iterator operator--(int) { iterator ci = *this; --i; return ci; }
      difference_type operator-(iterator const& o) const { return i - o.i; }
      iterator operator+(difference_type n) const { return iterator(i + n); }
      iterator operator-(difference_type n) const { return iterator(i - n); }
      bool operator<(iterator const& o) const { return i < o.i; }
      bool operator==(iterator const& ci) const { return i == ci.i; }
      bool operator!=(iterator const& ci) const { return i != ci.i; }
      T & operator *() const { return **i; }
      //    operator T *() const { return & **i; }
      //T *& get() { return *i; }
      T * operator->() const { return & (operator*()); }
      iterator & operator +=(difference_type d) { i += d; return *this; }
      iterator & operator -=(difference_type d) { i -= d; return *this; }
      reference operator[](difference_type d) const { return *iterator(i+d); } // for boost::iterator_range []
    private:
      pointer * i;
      friend class const_iterator;
      friend class OwnArray<T, MAX_SIZE, P>;
    };
    
    
    OwnArray();
    OwnArray(size_type);
    OwnArray(OwnArray const&);
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
    OwnArray(OwnArray&&);
#endif
    
    ~OwnArray();
    
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    size_type size() const;
    bool empty() const;
    reference operator[](size_type);
    const_reference operator[](size_type) const;
    
    self& operator=(self const&);
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
    self& operator=(self&&);
#endif
    
    
    void reserve(size_t){}
    size_type capacity() const { return MAX_SIZE;}
    template <typename D> void push_back(D*& d);
    template <typename D> void push_back(D* const& d);
    template <typename D> void push_back(std::auto_ptr<D> d);
    void push_back(T const& valueToCopy);
    bool is_back_safe() const;
    void pop_back();
    reference back();
    const_reference back() const;
    reference front();
    const_reference front() const;
    pointer const * data() const;
    void clear();
    iterator erase(iterator pos);
    iterator erase(iterator first, iterator last);
    template<typename S>
    void sort(S s);
    void sort();
    
    void swap(self& other);
    
    void fillView(ProductID const& id,
                  std::vector<void const*>& pointers,
                  helper_vector& helpers) const;
    
    void setPtr(std::type_info const& toType,
                unsigned long index,
                void const*& ptr) const;
    
    void fillPtrVector(std::type_info const& toType,
                       std::vector<unsigned long> const& indices,
                       std::vector<void const*>& ptrs) const;
    
    
    //Used by ROOT storage
    CMS_CLASS_VERSION(11)
    
    private:
    void destroy();
    template<typename O>
    struct Ordering {
      Ordering(O const& c) : comp(c) { }
      bool operator()(T const* t1, T const* t2) const {
        return comp(*t1, *t2);
      }
    private:
      O comp;
    };
    template<typename O>
    static Ordering<O> ordering(O const& comp) {
      return Ordering<O>(comp);
    }
    pointer data_[MAX_SIZE];
    size_type size_;
  };
  
  template<typename T, unsigned int M, typename P>
  inline OwnArray<T, M, P>::OwnArray() : data_{{0}}, size_(0) {
  }
  
  template<typename T, unsigned int M, typename P>
  inline OwnArray<T, M, P>::OwnArray(size_type n) : data_{{0}}, size_(n) {
  }
  
  template<typename T, unsigned int M, typename P>
  inline OwnArray<T, M, P>::OwnArray(OwnArray<T, M, P> const& o) : size_(o.size()) {
    size_type current = 0;
    for (const_iterator i = o.begin(), e = o.end(); i != e; ++i,++current)
      data_[current] = policy_type::clone(*i);
  }
  
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
  template<typename T, unsigned int M, typename P>
  inline OwnArray<T, M, P>::OwnArray(OwnArray<T, M, P>&& o)  {
    std::swap_ranges(data_,data_+M, o.data_);
  }
#endif
  
  template<typename T, unsigned int M, typename P>
  inline OwnArray<T, M, P>::~OwnArray() {
    destroy();
  }
  
  template<typename T, unsigned int M, typename P>
  inline OwnArray<T, M, P>& OwnArray<T, M, P>::operator=(OwnArray<T, M, P> const& o) {
    OwnArray<T,M,P> temp(o);
    swap(temp);
    return *this;
  }
  
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
  template<typename T, unsigned int M, typename P>
  inline OwnArray<T, M, P>& OwnArray<T, M, P>::operator=(OwnArray<T, M, P>&& o) {
    std::swap_ranges(data_,data_+M, o.data_);
    return *this;
  }
#endif
  
  
  template<typename T, unsigned int M, typename P>
  inline typename OwnArray<T, M, P>::iterator OwnArray<T, M, P>::begin() {
    return iterator(data_);
  }
  
   template<typename T, unsigned int M, typename P>
   inline typename OwnArray<T, M, P>::iterator OwnArray<T, M, P>::end() {
     return iterator(data_+size());
   }
  
  template<typename T, unsigned int M, typename P>
  inline typename OwnArray<T, M, P>::const_iterator OwnArray<T, M, P>::begin() const {
    return const_iterator(data_);
  }
  
  template<typename T, unsigned int M, typename P>
  inline typename OwnArray<T, M, P>::const_iterator OwnArray<T, M, P>::end() const {
    return const_iterator(data_+size());
  }
  
  template<typename T, unsigned int M, typename P>
  inline typename OwnArray<T, M, P>::size_type OwnArray<T, M, P>::size() const {
    return size_;
  }
  
  template<typename T, unsigned int M, typename P>
  inline bool OwnArray<T, M, P>::empty() const {
    return 0==size_;
  }

  template<typename T, unsigned int M, typename P>
  inline typename OwnArray<T, M, P>::reference OwnArray<T, M, P>::operator[](size_type n) {
    return *data_[n];
  }
  
  template<typename T, unsigned int M, typename P>
  inline typename OwnArray<T, M, P>::const_reference OwnArray<T, M, P>::operator[](size_type n) const {
    return *data_[n];
  }
  
  template<typename T, unsigned int M, typename P>
  template<typename D>
  inline void OwnArray<T, M, P>::push_back(D*& d) {
    // C++ does not yet support rvalue references, so d should only be
    // able to bind to an lvalue.
    // This should be called only for lvalues.
    data_[size_++]=d; 
    d = 0;
  }
  
  template<typename T, unsigned int M, typename P>
  template<typename D>
  inline void OwnArray<T, M, P>::push_back(D* const& d) {
    
    // C++ allows d to be bound to an lvalue or rvalue. But the other
    // signature should be a better match for an lvalue (because it
    // does not require an lvalue->rvalue conversion). Thus this
    // signature should only be chosen for rvalues.
    data_[size_++]=d; 
  }
  
  
  template<typename T, unsigned int M, typename P>
  template<typename D>
  inline void OwnArray<T, M, P>::push_back(std::auto_ptr<D> d) {
    data_[size_++]=d.release();
  }
  
  
  template<typename T, unsigned int M, typename P>
  inline void OwnArray<T, M, P>::push_back(T const& d) {
    data_[size_++]=policy_type::clone(d);
  }
  
  
  template<typename T, unsigned int M, typename P>
  inline void OwnArray<T, M, P>::pop_back() {
    // We have to delete the pointed-to thing, before we squeeze it
    // out of the vector...
    delete data_[--size_];
  }
  
  template <typename T, unsigned int M, typename P>
  inline bool OwnArray<T, M, P>::is_back_safe() const {
    return data_[size_-1] != 0;
  }
  
  template<typename T, unsigned int M, typename P>
  inline typename OwnArray<T, M, P>::reference OwnArray<T, M, P>::back() {
    pointer result = data_[size_-1];
    if (result == 0) {
      Exception::throwThis(errors::NullPointerError,
			   "In OwnArray::back() we have intercepted an attempt to dereference a null pointer\n"
			   "Since OwnArray is allowed to contain null pointers, you much assure that the\n"
			   "pointer at the end of the collection is not null before calling back()\n"
			   "if you wish to avoid this exception.\n"
			   "Consider using OwnArray::is_back_safe()\n");
    }
    return *result;
  }
  
  template<typename T, unsigned int M, typename P>
  inline typename OwnArray<T, M, P>::const_reference OwnArray<T, M, P>::back() const {
    pointer const * result = data_[size_-1];
    if (result == 0) {
      Exception::throwThis(errors::NullPointerError,
			   "In OwnArray::back() we have intercepted an attempt to dereference a null pointer\n"
			   "Since OwnArray is allowed to contain null pointers, you much assure that the\n"
			   "pointer at the end of the collection is not null before calling back()\n"
			   "if you wish to avoid this exception.\n"
			   "Consider using OwnArray::is_back_safe()\n");
    }
    return *result;
  }
  
  template<typename T, unsigned int M, typename P>
  inline typename OwnArray<T, M, P>::reference OwnArray<T, M, P>::front() {
    return *data_[0];
  }
  
  template<typename T, unsigned int M, typename P>
  inline typename OwnArray<T, M, P>::const_reference OwnArray<T, M, P>::front() const {
    return *data_[0];
  }
  
  template<typename T, unsigned int M, typename P>
  inline void OwnArray<T, M, P>::destroy() {
    pointer * b = data_, * e = data_+size();
    for(pointer * i = b; i != e; ++ i)
      delete * i;
  }
  
  template<typename T, unsigned int M, typename P>
  inline typename OwnArray<T, M, P>::pointer const * OwnArray<T, M, P>::data() const {
    return data_;
  }

   template<typename T, unsigned int M, typename P>
  inline void OwnArray<T, M, P>::clear() {
    destroy();
    size_=0;
  }

   template<typename T, unsigned int M, typename P>
  typename OwnArray<T, M, P>::iterator OwnArray<T, M, P>::erase(iterator pos) {
    pointer * b = pos.i;
    delete *b;
    pointer * e = data_+size();
    for(pointer * i = b; i != e-1; ++ i) *i = *(i+1);
    size_--;
    return iterator(b);
  }

   template<typename T, unsigned int M, typename P>
  typename OwnArray<T, M, P>::iterator OwnArray<T, M, P>::erase(iterator first, iterator last) {
     pointer * b = first.i, * e = last.i;
    for(pointer * i = b; i != e; ++ i) delete * i;
    pointer * l = data_+size();
    auto ib=b;
    for(pointer * i = e; i != l; ++i)  *(ib++) = *i;
    size_ -= (e-b);
    return iterator(b);
  }

   template<typename T, unsigned int M, typename P> template<typename S>
  void OwnArray<T, M, P>::sort(S comp) {
     std::sort(data_, data_+size(), ordering(comp));
  }

   template<typename T, unsigned int M, typename P>
  void OwnArray<T, M, P>::sort() {
    std::sort(data_, data_+size(), ordering(std::less<value_type>()));
  }

   template<typename T, unsigned int M, typename P>
  inline void OwnArray<T, M, P>::swap(OwnArray<T, M, P>& other) {
     std::swap_ranges(data_,data_+M, other.data_);
  }

   template<typename T, unsigned int M, typename P>
  void OwnArray<T, M, P>::fillView(ProductID const& id,
                                 std::vector<void const*>& pointers,
                                 helper_vector& helpers) const {
    typedef Ref<OwnArray>      ref_type ;
    typedef reftobase::RefHolder<ref_type> holder_type;

    size_type numElements = this->size();
    pointers.reserve(numElements);
    helpers.reserve(numElements);
    size_type key = 0;
    for(typename base::const_iterator i=data_.begin(), e=data_.end(); i!=e; ++i, ++key) {

      if (*i == 0) {
        Exception::throwThis(errors::NullPointerError,
          "In OwnArray::fillView() we have intercepted an attempt to put a null pointer\n"
          "into a View and that is not allowed.  It is probably an error that the null\n"
          "pointer was in the OwnArray in the first place.\n");
      }
      else {
        pointers.push_back(*i);
        holder_type h(ref_type(id, *i, key,this));
        helpers.push_back(&h);
      }
    }
  }

   template<typename T, unsigned int M, typename P>
  inline void swap(OwnArray<T, M, P>& a, OwnArray<T, M, P>& b) {
    a.swap(b);
  }

  //----------------------------------------------------------------------
  //
  // Free function template to support creation of Views.

  template <typename T, unsigned int M, typename P>
  inline
  void
  fillView(OwnArray<T,M,P> const& obj,
           ProductID const& id,
           std::vector<void const*>& pointers,
           helper_vector& helpers) {
    obj.fillView(id, pointers, helpers);
  }


  template <typename T, unsigned int M, typename P>
  struct has_fillView<edm::OwnArray<T, M, P> > {
    static bool const value = true;
  };


  // Free function templates to support the use of edm::Ptr.

  template <typename T, unsigned int M, typename P>
  inline
  void
  OwnArray<T,M,P>::setPtr(std::type_info const& toType,
                                   unsigned long index,
                                   void const*& ptr) const {
    detail::reallySetPtr<OwnArray<T,M,P> >(*this, toType, index, ptr);
  }

  template <typename T, unsigned int M, typename P>
  inline
  void
  setPtr(OwnArray<T,M,P> const& obj,
         std::type_info const& toType,
         unsigned long index,
         void const*& ptr) {
    obj.setPtr(toType, index, ptr);
  }

  template <typename T, unsigned int M, typename P>
  inline
  void
    OwnArray<T,M,P>::fillPtrVector(std::type_info const& toType,
                                  std::vector<unsigned long> const& indices,
                                  std::vector<void const*>& ptrs) const {
    detail::reallyfillPtrVector(*this, toType, indices, ptrs);
  }


  template <typename T, unsigned int M, typename P>
  inline
  void
  fillPtrVector(OwnArray<T,M,P> const& obj,
                std::type_info const& toType,
                std::vector<unsigned long> const& indices,
                std::vector<void const*>& ptrs) {
    obj.fillPtrVector(toType, indices, ptrs);
  }


  template <typename T, unsigned int M, typename P>
  struct has_setPtr<edm::OwnArray<T,M,P> > {
     static bool const value = true;
   };


}


#endif
