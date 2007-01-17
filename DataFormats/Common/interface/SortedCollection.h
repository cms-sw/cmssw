#ifndef Common_SortedCollection_h
#define Common_SortedCollection_h

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

$Id: SortedCollection.h,v 1.7 2007/01/11 23:39:17 paterno Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <vector>

#include "DataFormats/Common/interface/traits.h"

#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

  //------------------------------------------------------------
  // Forward declarations
  //

  template <class T> struct StrictWeakOrdering;  
  template <class T, class SORT = StrictWeakOrdering<T> > 
    class SortedCollection;
  
#if ! GCC_PREREQUISITE(3,4,4)
  //------------------------------------------------------------
  // The following template partial specialization can be removed
  // when we move to GCC 3.4.x
  //------------------------------------------------------------
  
  // Partial specialization of has_postinsert_trait template for any
  // SortedCollection<T>, regardless of T; all SortedCollection
  // classes have post_insert.
  
  template <class T, class SORT>
  struct has_postinsert_trait<edm::SortedCollection<T,SORT> >
  {
    static bool const value = true;
  };
#endif

  template <class T, class SORT>
  struct has_fillView<edm::SortedCollection<T,SORT> >
  {
    static bool const value = true;
  };


  template <class T>
  struct StrictWeakOrdering
  {
    typedef typename T::key_type key_type;

    // Each of the following comparisons are needed (at least with GCC's library).
    bool operator()(key_type a, T const& b) const { return a < b.id(); }
    bool operator()(T const& a, key_type b) const { return a.id() < b; }
    bool operator()(T const& a, T const& b) const { return a.id() < b.id(); }

    // This final comparison is not needed (at least with GCC's library).
    // bool operator()(key_type a, key_type b) const { return a < b; }
  };


  template <class T, class SORT>
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
    //template <class InputIterator>
    //SortedCollection(InputIterator b, InputIterator e);

    void push_back(T const& t);

    void swap(SortedCollection& other);

    void swap_contents(std::vector<T>& other);

    SortedCollection& operator=(SortedCollection const& rhs);

    bool empty() const;
    size_type size() const;
    size_type capacity() const;
    void reserve(size_type n);

    // Return a reference to the i'th item in the collection.
    // Not that the argument is an *integer*, not an object of
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

    void fillView(std::vector<void const*>& pointers) const;

  private:

    typedef std::vector<T> collection_type;
    typedef typename collection_type::const_iterator const_inner_iterator;
    typedef typename collection_type::iterator       inner_iterator;

    collection_type   obj;
  };

  template <class T, class SORT>
  inline
  SortedCollection<T,SORT>::SortedCollection() : obj() {}

  template <class T, class SORT>
  inline
  SortedCollection<T,SORT>::SortedCollection(size_type n) : obj(n) {}

  template <class T, class SORT>
  inline
  SortedCollection<T,SORT>::SortedCollection(std::vector<T> const& vec) : obj(vec) {}

  template <class T, class SORT>
  inline
  SortedCollection<T,SORT>::SortedCollection(SortedCollection<T,SORT> const& h) : obj(h.obj) {}

  template <class T, class SORT>
  inline
  void
  SortedCollection<T,SORT>::push_back(T const& t) {
    obj.push_back(t);
  }

  template <class T, class SORT>
  inline
  void
  SortedCollection<T,SORT>::swap(SortedCollection<T,SORT>& other) {
    obj.swap(other.obj);
  }

  template <class T, class SORT>
  inline
  void
  SortedCollection<T,SORT>::swap_contents(std::vector<T>& other) {
    obj.swap(other);
  }

  template <class T, class SORT>
  inline
  SortedCollection<T,SORT>&
  SortedCollection<T,SORT>::operator=(SortedCollection<T,SORT> const& rhs) {
    SortedCollection<T,SORT> temp(rhs);
    this->swap(temp);
    return *this;
  }

  template <class T, class SORT>
  inline
  bool
  SortedCollection<T,SORT>::empty() const {
    return obj.empty();
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::size_type
  SortedCollection<T,SORT>::size() const {
    return obj.size();
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::size_type
  SortedCollection<T,SORT>::capacity() const {
    return obj.capacity();
  }

  template <class T, class SORT>
  inline
  void
  SortedCollection<T,SORT>::reserve(typename SortedCollection<T,SORT>::size_type n) {
    obj.reserve(n);
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::reference
  SortedCollection<T,SORT>::operator[](size_type i) {
    return obj[i];
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::const_reference
  SortedCollection<T,SORT>::operator[](size_type i) const {
    return obj[i];
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::iterator
  SortedCollection<T,SORT>::find(key_type key)
  {
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

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::const_iterator
  SortedCollection<T,SORT>::find(key_type key) const
  {
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

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::const_iterator
  SortedCollection<T,SORT>::begin() const {
    return obj.begin();
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::const_iterator
  SortedCollection<T,SORT>::end() const {
    return obj.end();
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::iterator
  SortedCollection<T,SORT>::begin()
  {
    return obj.begin();
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::iterator
  SortedCollection<T,SORT>::end()
  {
    return obj.end();
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::const_reference
  SortedCollection<T,SORT>::front() const
  {
    return obj.front();
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::reference
  SortedCollection<T,SORT>::front()
  {
    return obj.front();
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::const_reference
  SortedCollection<T,SORT>::back() const
  {
    return obj.back();
  }

  template <class T, class SORT>
  inline
  typename SortedCollection<T,SORT>::reference
  SortedCollection<T,SORT>::back()
  {
    return obj.back();
  }

  template <class T, class SORT>
  inline
  void
  SortedCollection<T,SORT>::sort()
  {
    key_compare  comp;
    std::sort(obj.begin(), obj.end(), comp);
  }

  template <class T, class SORT>
  inline
  void
  SortedCollection<T,SORT>::post_insert()
  {
    // After insertion, we make sure our contents are sorted.
    sort();
  }

  template <class T, class SORT>
  void
  SortedCollection<T,SORT>::fillView(std::vector<void const*>& pointers) const
  {
    pointers.reserve(this->size());
    for(const_iterator i=begin(), e=end(); i!=e; ++i)
      pointers.push_back(&(*i));
  }

  // Free swap function
  template <class T, class SORT>
  inline
  void
  swap(SortedCollection<T,SORT>& a, SortedCollection<T,SORT>& b) 
  {
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
  
  template <class T, class SORT, class ALLOC>
  inline
  bool 
  operator== (SortedCollection<T, SORT> const& c,
	      std::vector<T, ALLOC>     const& v)
  {
    return c.size() == v.size() && std::equal(v.begin(), v.end(), c.begin());
  }

  // comparison of two SortedCollections is done by comparing the
  // collected elements, in order for equality.
  // operator==(T const& a, T const& b) is used to compare the elements.

  template <class T, class SORT>
  inline
  bool
  operator==(SortedCollection<T, SORT> const& a,
	     SortedCollection<T, SORT> const& b)
  {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());    
  }

  //----------------------------------------------------------------------
  //
  // Free function template to support creation of Views.

  template <class T, class SORT>
  inline
  void
  fillView(SortedCollection<T,SORT> const& obj,
	   std::vector<void const*>& pointers)
  {
    obj.fillView(pointers);
  }

#if ! GCC_PREREQUISITE(3,4,4)
  // has swap function
  template <class T, class SORT>
  struct has_swap<edm::SortedCollection<T,SORT> > {
    static bool const value = true;
  };
#endif
}

#endif
