#ifndef DataFormats_Common_SortedCollection_h
#define DataFormats_Common_SortedCollection_h

/*----------------------------------------------------------------------

SortedCollection: A collection of homogeneous objects that can be used
for an EDProduct. SortedCollection is *not* designed for use as a base
class (it has no virtual functions).

SortedCollection is in some ways similar to a sequence
(e.g. std::vector and std::list), and in other ways is similar to an
associative collection (e.g. std::map and std::set). SortedCollection
provides keyed access (via operator[]() and find()) to its contained
values. In normal usage, the values contained in a SortedCollection
are sorted according to the order imposed by the ordering of the keys.

CAVEATS:

1. The user must make sure that two VALUES with the same KEY are not
never inserted into the sequence. The SortedCollection does not
prevent this, nor does it detect this. However, 'find' will be
unreliable if such duplicate entries are made.

**************** Much more is needed here! ****************

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/fillPtrVector.h"
#include "DataFormats/Common/interface/FillView.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

#include <algorithm>
#include <typeinfo>
#include <vector>

namespace edm {

  //------------------------------------------------------------
  // Forward declarations
  //

  template<typename T> struct StrictWeakOrdering;
  template<typename T, typename SORT = StrictWeakOrdering<T> >
    class SortedCollection;

  template<typename T, typename SORT>
  struct has_fillView<edm::SortedCollection<T, SORT> > {
    static bool const value = true;
  };

  template<typename T, typename SORT>
  struct has_setPtr<edm::SortedCollection<T, SORT> > {
    static bool const value = true;
  };

  template<typename T>
  struct StrictWeakOrdering {
    typedef typename T::key_type key_type;

    // Each of the following comparisons are needed (at least with GCC's library).
    bool operator()(key_type a, T const& b) const { return a < b.id(); }
    bool operator()(T const& a, key_type b) const { return a.id() < b; }
    bool operator()(T const& a, T const& b) const { return a.id() < b.id(); }

    // This final comparison is not needed (at least with GCC's library).
    // bool operator()(key_type a, key_type b) const { return a < b; }
  };


  template<typename T, typename SORT>
  class SortedCollection {
  public:
    typedef T    value_type;    // the values we contain
    typedef SORT key_compare;   // function object for sorting

    typedef typename std::vector<T>::const_iterator  const_iterator;
    typedef typename std::vector<T>::iterator        iterator;
    typedef typename std::vector<T>::const_reference const_reference;
    typedef typename std::vector<T>::reference       reference;

    typedef typename std::vector<T>::size_type      size_type;

    // This needs to be turned into a template parameter, perhaps with
    // a default --- if there is a way to slip in the default without
    // growing any dependence on the code supplying the key!
    typedef typename key_compare::key_type key_type;

    SortedCollection();
    explicit SortedCollection(size_type n);
    explicit SortedCollection(std::vector<T> const& vec);
    SortedCollection(SortedCollection const& h);

    // Add the following when needed
    //template<typename InputIterator>
    //SortedCollection(InputIterator b, InputIterator e);

    void push_back(T const& t);
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
    void push_back(T && t) { obj.push_back(t);}

    template<typename... Args >
    void emplace_back( Args&&... args ) { obj.emplace_back(args...);}
#endif
    void pop_back() { obj.pop_back(); }

    void swap(SortedCollection& other);

    void swap_contents(std::vector<T>& other);

    SortedCollection& operator=(SortedCollection const& rhs);

    bool empty() const;
    size_type size() const;
    size_type capacity() const;
    void reserve(size_type n);

    // Return a reference to the i'th item in the collection.
    // Note that the argument is an *integer*, not an object of
    //   type key_type
    reference       operator[](size_type i);
    const_reference operator[](size_type i) const;

    // Find the item with key matching k. If no such item is found,
    // return end();
    iterator       find(key_type k);
    const_iterator find(key_type k) const;

    const_iterator begin() const;
    const_iterator end() const;

    iterator begin();
    iterator end();

    const_reference front() const;
    reference       front();
    const_reference back() const;
    reference       back();

    // Sort the elements of the vector, in the order determined by the
    // keys. Note that the Event will make sure to call this function
    // after the SortedCollection has been put into the Event, so
    // there is no need to call it in user code (unless one needs the
    // collection sorted *before* it is inserted into the Event).
    void sort();

    // This function will be called by the edm::Event after the
    // SortedCollection has been inserted into the Event.
    void post_insert();

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
    CMS_CLASS_VERSION(10)

  private:

    typedef std::vector<T> collection_type;
    typedef typename collection_type::const_iterator const_inner_iterator;
    typedef typename collection_type::iterator       inner_iterator;

    collection_type   obj;
  };

  template<typename T, typename SORT>
  inline
  SortedCollection<T, SORT>::SortedCollection() : obj() {}

  template<typename T, typename SORT>
  inline
  SortedCollection<T, SORT>::SortedCollection(size_type n) : obj(n) {}

  template<typename T, typename SORT>
  inline
  SortedCollection<T, SORT>::SortedCollection(std::vector<T> const& vec) : obj(vec) {}

  template<typename T, typename SORT>
  inline
  SortedCollection<T, SORT>::SortedCollection(SortedCollection<T, SORT> const& h) : obj(h.obj) {}

  template<typename T, typename SORT>
  inline
  void
  SortedCollection<T, SORT>::push_back(T const& t) {
    obj.push_back(t);
  }

  template<typename T, typename SORT>
  inline
  void
  SortedCollection<T, SORT>::swap(SortedCollection<T, SORT>& other) {
    obj.swap(other.obj);
  }

  template<typename T, typename SORT>
  inline
  void
  SortedCollection<T, SORT>::swap_contents(std::vector<T>& other) {
    obj.swap(other);
  }

  template<typename T, typename SORT>
  inline
  SortedCollection<T, SORT>&
  SortedCollection<T, SORT>::operator=(SortedCollection<T, SORT> const& rhs) {
    SortedCollection<T, SORT> temp(rhs);
    this->swap(temp);
    return *this;
  }

  template<typename T, typename SORT>
  inline
  bool
  SortedCollection<T, SORT>::empty() const {
    return obj.empty();
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::size_type
  SortedCollection<T, SORT>::size() const {
    return obj.size();
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::size_type
  SortedCollection<T, SORT>::capacity() const {
    return obj.capacity();
  }

  template<typename T, typename SORT>
  inline
  void
  SortedCollection<T, SORT>::reserve(typename SortedCollection<T, SORT>::size_type n) {
    obj.reserve(n);
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::reference
  SortedCollection<T, SORT>::operator[](size_type i) {
    return obj[i];
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::const_reference
  SortedCollection<T, SORT>::operator[](size_type i) const {
    return obj[i];
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::iterator
  SortedCollection<T, SORT>::find(key_type key) {
    // This fails if the SortedCollection has not been sorted. It is
    // up to the user (with the help of the Event) to make sure this
    // has been done.
    key_compare comp;
    inner_iterator last = obj.end();
    inner_iterator loc = std::lower_bound(obj.begin(),
                                          last,
                                          key,
                                          comp);
    return loc == last || comp(key, *loc) ? last : loc;
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::const_iterator
  SortedCollection<T, SORT>::find(key_type key) const {
    // This fails if the SortedCollection has not been sorted. It is
    // up to the user (with the help of the Event) to make sure this
    // has been done.
    key_compare  comp;
    const_inner_iterator last = obj.end();
    const_inner_iterator loc = std::lower_bound(obj.begin(),
                                                last,
                                                key,
                                                comp);
    return loc == last || comp(key, *loc) ? last : loc;
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::const_iterator
  SortedCollection<T, SORT>::begin() const {
    return obj.begin();
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::const_iterator
  SortedCollection<T, SORT>::end() const {
    return obj.end();
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::iterator
  SortedCollection<T, SORT>::begin() {
    return obj.begin();
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::iterator
  SortedCollection<T, SORT>::end() {
    return obj.end();
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::const_reference
  SortedCollection<T, SORT>::front() const {
    return obj.front();
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::reference
  SortedCollection<T, SORT>::front() {
    return obj.front();
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::const_reference
  SortedCollection<T, SORT>::back() const {
    return obj.back();
  }

  template<typename T, typename SORT>
  inline
  typename SortedCollection<T, SORT>::reference
  SortedCollection<T, SORT>::back() {
    return obj.back();
  }

  template<typename T, typename SORT>
  inline
  void
  SortedCollection<T, SORT>::sort() {
    key_compare  comp;
    std::sort(obj.begin(), obj.end(), comp);
  }

  template<typename T, typename SORT>
  inline
  void
  SortedCollection<T, SORT>::post_insert() {
    // After insertion, we make sure our contents are sorted.
    sort();
  }

  template<typename T, typename SORT>
  inline
  void
  SortedCollection<T, SORT>::fillView(ProductID const& id,
                                     std::vector<void const*>& pointers,
                                     helper_vector& helpers) const {
    detail::reallyFillView(*this, id, pointers, helpers);
  }

  template<typename T, typename SORT>
  inline
  void
  SortedCollection<T, SORT>::setPtr(std::type_info const& toType,
                                   unsigned long index,
                                   void const*& ptr) const {
    detail::reallySetPtr(*this, toType, index, ptr);
  }

  template<typename T, typename SORT>
  inline
  void
  SortedCollection<T, SORT>::fillPtrVector(std::type_info const& toType,
                                          std::vector<unsigned long> const& indices,
                                          std::vector<void const*>& ptrs) const {
    detail::reallyfillPtrVector(*this, toType, indices, ptrs);
  }

  // Free swap function
  template<typename T, typename SORT>
  inline
  void
  swap(SortedCollection<T, SORT>& a, SortedCollection<T, SORT>& b) {
    a.swap(b);
  }

  //----------------------------------------------------------------------
  //
  // Free function templates to support comparisons.

  // The two function templates below are not written as a single
  // function template in order to avoid inadvertent matches with
  // inappropriate template arguments. Template template parameters
  // were avoided to stay away from generic programming problems
  // sometimes encountered in the presence of such parameters.  If
  // support for comparison between SortedCollections and other sorts
  // of collections is needed, it can be added.

  // comparison with vector tests to see whether the entries in the
  // SortedCollection are the same as the entries in the vector, *and
  // in the same order.
  // operator==(T const& a, T const& b) is used to compare the elements in
  // the collections.

  template<typename T, typename SORT, typename ALLOC>
  inline
  bool
  operator==(SortedCollection<T, SORT> const& c,
             std::vector<T, ALLOC>     const& v) {
    return c.size() == v.size() && std::equal(v.begin(), v.end(), c.begin());
  }

  // comparison of two SortedCollections is done by comparing the
  // collected elements, in order for equality.
  // operator==(T const& a, T const& b) is used to compare the elements.

  template<typename T, typename SORT>
  inline
  bool
  operator==(SortedCollection<T, SORT> const& a,
             SortedCollection<T, SORT> const& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
  }

  //----------------------------------------------------------------------
  //
  // Free function template to support creation of Views.

  template<typename T, typename SORT>
  inline
  void
  fillView(SortedCollection<T, SORT> const& obj,
           ProductID const& id,
           std::vector<void const*>& pointers,
           helper_vector& helpers) {
    obj.fillView(id, pointers, helpers);
  }

  // Free function templates to support the use of edm::Ptr.
  template<typename T, typename SORT>
  inline
  void
  setPtr(SortedCollection<T, SORT> const& obj,
         std::type_info const& toType,
         unsigned long index,
         void const*& ptr) {
    obj.setPtr(toType, index, ptr);
  }

  template<typename T, typename SORT>
  inline
  void
  fillPtrVector(SortedCollection<T, SORT> const& obj,
                std::type_info const& toType,
                std::vector<unsigned long> const& indices,
                std::vector<void const*>& ptrs) {
    obj.fillPtrVector(toType, indices, ptrs);
  }
}

#endif
