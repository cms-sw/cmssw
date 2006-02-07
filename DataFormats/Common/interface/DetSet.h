#ifndef Common_DetSet_h
#define Common_DetSet_h

/*----------------------------------------------------------------------
  
DetSet: A struct which combines a collection of homogeneous objects
associated with a common DetId with a DetId instance, holding the
common DetId value. The collected objects may or may not contain their
own copy of the common DetId.

$Id: DetSet.h,v 1.1 2006/01/27 21:20:04 paterno Exp $

----------------------------------------------------------------------*/

#include "boost/cstdint.hpp"
#include <vector>

namespace edm {

  
  typedef uint32_t det_id_type;

  template <class T>
  struct DetSet
  {
    typedef std::vector<T>                           collection_type;
    // We don't just use T as value-type, in case we switch to a
    // fancier underlying container.
    typedef typename collection_type::value_type     value_type;
    typedef typename collection_type::iterator       iterator;
    typedef typename collection_type::const_iterator const_iterator;

    DetSet() : id(0), data() { }
    explicit DetSet(det_id_type i) : id(i), data() { }

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
  operator< (DetSet<T> const& x, DetSet<T> const& y)
  {
    return x.id < y.id;
  }

  template <class T>
  inline
  bool
  operator< (DetSet<T> const& x, det_id_type y)
  {
    return x.id < y;
  }

  template <class T>
  inline
  bool
  operator< (det_id_type x, DetSet<T> const& y)
  {
    return x < y.id;
  }

} // namespace edm;

#endif
