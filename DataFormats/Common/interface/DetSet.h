#ifndef DataFormats_Common_DetSet_h
#define DataFormats_Common_DetSet_h

/*----------------------------------------------------------------------
  
DetSet: A struct which combines a collection of homogeneous objects
associated with a common DetId with a DetId instance, holding the
common DetId value. The collected objects may or may not contain their
own copy of the common DetId.

$Id: DetSet.h,v 1.14 2011/03/08 14:01:16 innocent Exp $

----------------------------------------------------------------------*/

#include <vector>
#include <stdint.h>
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

namespace edm {
  typedef uint32_t det_id_type;

  template <class T>
  struct DetSet
  {
    typedef std::vector<T>                           collection_type;
    // We don't just use T as value-type, in case we switch to a
    // fancier underlying container.
    typedef typename collection_type::value_type      value_type;
    typedef typename collection_type::reference       reference;
    typedef typename collection_type::const_reference const_reference;
    typedef typename collection_type::iterator        iterator;
    typedef typename collection_type::const_iterator  const_iterator;
    typedef typename collection_type::size_type       size_type;

    /// default constructor
    DetSet() : id(0), data() { }
    /// constructor by detector identifier
    explicit DetSet(det_id_type i) : id(i), data() { }

#if defined( __GXX_EXPERIMENTAL_CXX0X__)
    DetSet(DetSet<T> const & rh) : id(rh.id), data(rh.data){}

    DetSet<T> & operator=(DetSet<T> const & rh)  {
      id = rh.id;
      data = rh.data;
      return * this;
    }

    DetSet(DetSet<T> && rh) : id(rh.id), data(std::move(rh.data)){}
    
    DetSet<T> & operator=(DetSet<T> && rh)  {
      id = rh.id;
      data.swap(rh.data);
      return * this;
    }
#endif

    iterator begin() { return data.begin(); }
    iterator end() { return data.end(); }
    const_iterator begin() const { return data.begin(); }
    const_iterator end() const { return data.end(); }
    size_type size() const { return data.size(); }
    bool empty() const { return data.empty(); }
    reference operator[](size_type i) { return data[ i ]; }
    const_reference operator[](size_type i) const { return data[ i ]; }
    void reserve(size_t s) { data.reserve(s); }
    void push_back(const T & t) { data.push_back(t); }
    void clear() { data.clear(); }
    void swap(DetSet<T> & other);

    det_id_type detId() const { return id; }

    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

    det_id_type      id;
    collection_type  data;
  }; // template struct DetSet


  // TODO: If it turns out that it is confusing to have operator<
  // defined for DetSet, because this op< ignores the data member
  // 'data', we could instead specialize the std::less class template
  // directly.

  template <class T>
  inline
  bool
  operator< (DetSet<T> const& x, DetSet<T> const& y) {
    return x.detId() < y.detId();
  }

  template <class T>
  inline
  bool
  operator< (DetSet<T> const& x, det_id_type y) {
    return x.detId() < y;
  }

  template <class T>
  inline
  bool
  operator< (det_id_type x, DetSet<T> const& y) {
    return x < y.detId();
  }

  template <class T>
  inline
  void
  DetSet<T>::swap(DetSet<T> & other) {
    data.swap(other.data);
    std::swap(id, other.id);
  }

  template <class T>
  inline
  void
  swap(DetSet<T> & a, DetSet<T> & b) {
    a.swap(b);
  }

} // namespace edm;

#endif
