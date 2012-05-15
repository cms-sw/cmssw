#ifndef DataFormats_Common_OwnVector_h
#define DataFormats_Common_OwnVector_h

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
  template <typename T, typename P = ClonePolicy<T> >
  class OwnVector {
  private:
#if defined(CMS_USE_DEBUGGING_ALLOCATOR)
    typedef std::vector<T*, debugging_allocator<T> > base;
#else
    typedef std::vector<T*> base;
#endif

  public:
    typedef typename base::size_type size_type;
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
      const_iterator(typename base::const_iterator const& it) : i(it) { }
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
      typename base::const_iterator i;
    };
    class iterator {
    public:
      typedef T value_type;
      typedef T * pointer;
      typedef T & reference;
      typedef ptrdiff_t difference_type;
      typedef typename base::iterator::iterator_category iterator_category;
      iterator(typename base::iterator const& it) : i(it) { }
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
      typename base::iterator i;
      friend class const_iterator;
      friend class OwnVector<T, P>;
   };


    OwnVector();
    OwnVector(size_type);
    OwnVector(OwnVector const&);
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
    OwnVector(OwnVector&&);
#endif

    ~OwnVector();

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    size_type size() const;
    bool empty() const;
    reference operator[](size_type);
    const_reference operator[](size_type) const;

    OwnVector<T, P>& operator=(OwnVector<T, P> const&);
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
    OwnVector<T, P>& operator=(OwnVector<T, P>&&);
#endif


    void reserve(size_t);
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
    base const& data() const;
    void clear();
    iterator erase(iterator pos);
    iterator erase(iterator first, iterator last);
    template<typename S>
    void sort(S s);
    void sort();

    void swap(OwnVector<T, P>& other);

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
    base data_;
  };

  template<typename T, typename P>
  inline OwnVector<T, P>::OwnVector() : data_() {
  }

  template<typename T, typename P>
  inline OwnVector<T, P>::OwnVector(size_type n) : data_(n) {
  }

  template<typename T, typename P>
  inline OwnVector<T, P>::OwnVector(OwnVector<T, P> const& o) : data_(o.size()) {
    size_type current = 0;
    for (const_iterator i = o.begin(), e = o.end(); i != e; ++i,++current)
      data_[current] = policy_type::clone(*i);
  }

#if defined(__GXX_EXPERIMENTAL_CXX0X__)
  template<typename T, typename P>
  inline OwnVector<T, P>::OwnVector(OwnVector<T, P>&& o)  {
    data_.swap(o.data_);
  }
#endif

  template<typename T, typename P>
  inline OwnVector<T, P>::~OwnVector() {
    destroy();
  }

  template<typename T, typename P>
  inline OwnVector<T, P>& OwnVector<T, P>::operator=(OwnVector<T, P> const& o) {
    OwnVector<T,P> temp(o);
    swap(temp);
    return *this;
  }

#if defined(__GXX_EXPERIMENTAL_CXX0X__)
  template<typename T, typename P>
  inline OwnVector<T, P>& OwnVector<T, P>::operator=(OwnVector<T, P>&& o) {
    data_.swap(o.data_);
    return *this;
  }
#endif


  template<typename T, typename P>
  inline typename OwnVector<T, P>::iterator OwnVector<T, P>::begin() {
    return iterator(data_.begin());
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::iterator OwnVector<T, P>::end() {
    return iterator(data_.end());
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::const_iterator OwnVector<T, P>::begin() const {
    return const_iterator(data_.begin());
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::const_iterator OwnVector<T, P>::end() const {
    return const_iterator(data_.end());
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::size_type OwnVector<T, P>::size() const {
    return data_.size();
  }

  template<typename T, typename P>
  inline bool OwnVector<T, P>::empty() const {
    return data_.empty();
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::reference OwnVector<T, P>::operator[](size_type n) {
    return *data_[n];
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::const_reference OwnVector<T, P>::operator[](size_type n) const {
    return *data_[n];
  }

  template<typename T, typename P>
  inline void OwnVector<T, P>::reserve(size_t n) {
    data_.reserve(n);
  }

  template<typename T, typename P>
  template<typename D>
  inline void OwnVector<T, P>::push_back(D*& d) {
    // C++ does not yet support rvalue references, so d should only be
    // able to bind to an lvalue.
    // This should be called only for lvalues.
    data_.push_back(d);
    d = 0;
  }

  template<typename T, typename P>
  template<typename D>
  inline void OwnVector<T, P>::push_back(D* const& d) {

    // C++ allows d to be bound to an lvalue or rvalue. But the other
    // signature should be a better match for an lvalue (because it
    // does not require an lvalue->rvalue conversion). Thus this
    // signature should only be chosen for rvalues.
    data_.push_back(d);
  }


  template<typename T, typename P>
  template<typename D>
  inline void OwnVector<T, P>::push_back(std::auto_ptr<D> d) {
    data_.push_back(d.release());
  }


  template<typename T, typename P>
  inline void OwnVector<T, P>::push_back(T const& d) {
    data_.push_back(policy_type::clone(d));
  }


  template<typename T, typename P>
  inline void OwnVector<T, P>::pop_back() {
    // We have to delete the pointed-to thing, before we squeeze it
    // out of the vector...
    delete data_.back();
    data_.pop_back();
  }

  template <typename T, typename P>
  inline bool OwnVector<T, P>::is_back_safe() const {
    return data_.back() != 0;
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::reference OwnVector<T, P>::back() {
    T* result = data_.back();
    if (result == 0) {
      Exception::throwThis(errors::NullPointerError,
        "In OwnVector::back() we have intercepted an attempt to dereference a null pointer\n"
        "Since OwnVector is allowed to contain null pointers, you much assure that the\n"
        "pointer at the end of the collection is not null before calling back()\n"
        "if you wish to avoid this exception.\n"
        "Consider using OwnVector::is_back_safe()\n");
    }
    return *data_.back();
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::const_reference OwnVector<T, P>::back() const {
    T* result = data_.back();
    if (result == 0) {
      Exception::throwThis(errors::NullPointerError,
        "In OwnVector::back() we have intercepted an attempt to dereference a null pointer\n"
        "Since OwnVector is allowed to contain null pointers, you much assure that the\n"
        "pointer at the end of the collection is not null before calling back()\n"
        "if you wish to avoid this exception.\n"
        "Consider using OwnVector::is_back_safe()\n");
    }
    return *data_.back();
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::reference OwnVector<T, P>::front() {
    return *data_.front();
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::const_reference OwnVector<T, P>::front() const {
    return *data_.front();
  }

  template<typename T, typename P>
  inline void OwnVector<T, P>::destroy() {
    typename base::const_iterator b = data_.begin(), e = data_.end();
    for(typename base::const_iterator i = b; i != e; ++ i)
      delete * i;
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::base const& OwnVector<T, P>::data() const {
    return data_;
  }

  template<typename T, typename P>
  inline void OwnVector<T, P>::clear() {
    destroy();
    data_.clear();
  }

  template<typename T, typename P>
  typename OwnVector<T, P>::iterator OwnVector<T, P>::erase(iterator pos) {
    delete * pos.i;
    return iterator(data_.erase(pos.i));
  }

  template<typename T, typename P>
  typename OwnVector<T, P>::iterator OwnVector<T, P>::erase(iterator first, iterator last) {
    typename base::iterator b = first.i, e = last.i;
    for(typename base::iterator i = b; i != e; ++ i)
      delete * i;
    return iterator(data_.erase(b, e));
  }

  template<typename T, typename P> template<typename S>
  void OwnVector<T, P>::sort(S comp) {
    std::sort(data_.begin(), data_.end(), ordering(comp));
  }

  template<typename T, typename P>
  void OwnVector<T, P>::sort() {
    std::sort(data_.begin(), data_.end(), ordering(std::less<value_type>()));
  }

  template<typename T, typename P>
  inline void OwnVector<T, P>::swap(OwnVector<T, P>& other) {
    data_.swap(other.data_);
  }

  template<typename T, typename P>
  void OwnVector<T, P>::fillView(ProductID const& id,
                                 std::vector<void const*>& pointers,
                                 helper_vector& helpers) const {
    typedef Ref<OwnVector>      ref_type ;
    typedef reftobase::RefHolder<ref_type> holder_type;

    size_type numElements = this->size();
    pointers.reserve(numElements);
    helpers.reserve(numElements);
    size_type key = 0;
    for(typename base::const_iterator i=data_.begin(), e=data_.end(); i!=e; ++i, ++key) {

      if (*i == 0) {
        Exception::throwThis(errors::NullPointerError,
          "In OwnVector::fillView() we have intercepted an attempt to put a null pointer\n"
          "into a View and that is not allowed.  It is probably an error that the null\n"
          "pointer was in the OwnVector in the first place.\n");
      }
      else {
        pointers.push_back(*i);
        holder_type h(ref_type(id, *i, key,this));
        helpers.push_back(&h);
      }
    }
  }

  template<typename T, typename P>
  inline void swap(OwnVector<T, P>& a, OwnVector<T, P>& b) {
    a.swap(b);
  }

  //----------------------------------------------------------------------
  //
  // Free function template to support creation of Views.

  template <typename T, typename P>
  inline
  void
  fillView(OwnVector<T,P> const& obj,
           ProductID const& id,
           std::vector<void const*>& pointers,
           helper_vector& helpers) {
    obj.fillView(id, pointers, helpers);
  }


  template <typename T, typename P>
  struct has_fillView<edm::OwnVector<T, P> > {
    static bool const value = true;
  };


  // Free function templates to support the use of edm::Ptr.

  template <typename T, typename P>
  inline
  void
  OwnVector<T,P>::setPtr(std::type_info const& toType,
                                   unsigned long index,
                                   void const*& ptr) const {
    detail::reallySetPtr<OwnVector<T,P> >(*this, toType, index, ptr);
  }

  template <typename T, typename P>
  inline
  void
  setPtr(OwnVector<T,P> const& obj,
         std::type_info const& toType,
         unsigned long index,
         void const*& ptr) {
    obj.setPtr(toType, index, ptr);
  }

  template <typename T, typename P>
  inline
  void
    OwnVector<T,P>::fillPtrVector(std::type_info const& toType,
                                  std::vector<unsigned long> const& indices,
                                  std::vector<void const*>& ptrs) const {
    detail::reallyfillPtrVector(*this, toType, indices, ptrs);
  }


  template <typename T, typename P>
  inline
  void
  fillPtrVector(OwnVector<T,P> const& obj,
                std::type_info const& toType,
                std::vector<unsigned long> const& indices,
                std::vector<void const*>& ptrs) {
    obj.fillPtrVector(toType, indices, ptrs);
  }


  template <typename T, typename P>
   struct has_setPtr<edm::OwnVector<T,P> > {
     static bool const value = true;
   };


}


#endif
