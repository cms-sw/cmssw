#ifndef Common_DetSetVector_h
#define Common_DetSetVector_h

/*----------------------------------------------------------------------
  
DetSetVector: A collection of homogeneous objects that can be used for
an EDProduct. DetSetVector is *not* designed for use as a base class
(it has no virtual functions).

DetSetVector<T> contains a vector<DetSet<T> >, sorted on DetId, and
provides fast (O(log n)) lookups, but only O(n) insertion.

It provides an interface such that EdmRef2 can directly reference, and
provide access to, individual T objects.

The collection appears to the user as if it were a sequence of
DetSet<T>; e.g., operator[] returns a DetSet<T>&. However, the
argument to operator[] specifies the (DetId) identifier of the vector
to be returned, *not* the ordinal number of the T to be returned.

			  ------------------
   It is critical that users DO NOT MODIFY the id data member of a
   DetSet object in a DetSetVector.
			  ------------------

$Id: DetSetVector.h,v 1.1 2006/01/27 21:20:04 paterno Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <vector>

#include "boost/concept_check.hpp"

#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  //------------------------------------------------------------
  // Forward declarations
  template <class T> class DetSetVector;

  //------------------------------------------------------------
  // The following template partial specialization can be removed
  // when we move to GCC 3.4.x
  //------------------------------------------------------------
  
  // Partial specialization of has_postinsert_trait template for any
  // DetSetVector<T>, regardless of T; all DetSetVector classes have
  // post_insert.
  
  template <class T>
  struct has_postinsert_trait<edm::DetSetVector<T> >
  {
    static bool const value = true;
  };

  //------------------------------------------------------------
  // Helper function, to regularize throwing of exceptions.
  //------------------------------------------------------------

  namespace detail
  {
    // Throw an edm::Exception with an appropriate message
    inline
    void _throw_range(det_id_type i)
    {
      throw edm::Exception(errors::InvalidReference)
	<< "DetSetVector::operator[] called with index not in collection;\n"
	<< "index value: " << i;
    }
  }

  //------------------------------------------------------------
  //

  template <class T>
  class DetSetVector 
  {
    /// DetSetVector requires that T objects can be compared with
    /// operator<.
    BOOST_CLASS_REQUIRE(T, boost, LessThanComparableConcept);
  public:

    typedef T                   value_type;
    typedef DetSet<T>           detset;
    typedef std::vector<detset> collection_type;

    typedef detset&        reference;
    typedef detset const&  const_reference;

    typedef typename collection_type::iterator       iterator;
    typedef typename collection_type::const_iterator const_iterator;
    typedef typename collection_type::size_type      size_type;

    /// Compiler-generated default c'tor, copy c'tor, d'tor and
    /// assignment are correct.

    // Add the following only if needed.
    //template <class InputIterator>
    //DetSetVector(InputIterator b, InputIterator e);

    void swap(DetSetVector& other);

    //    DetSetVector& operator=(DetSetVector const& rhs);

    ///  Insert the given DetSet.
    // What should happen if there is already a DetSet with this
    // DetId? Right now, it is up to the user *not* to do this. If you
    // are unsure whether or not your DetId is already in the
    // DetSetVector, then use 'find_or_insert(id)' instead.
    void insert(detset const& s);

    /// Find the DetSet with the given DetId, and return a reference
    /// to it. If there is none, create one with the right DetId, and
    /// an empty vector, and return a reference to the new one.
    reference find_or_insert(det_id_type id);      

    /// Return true if we contain no DetSets.
    bool empty() const;

    /// Return the number of contained DetSets
    size_type size() const;

    // Do we need a short-hand method to return the number of T
    // instances? If so, do we optimize for size (calculate on the
    // fly) or speed (keep a current cache)?

    /// Return an iterator to the DetSet with the given id, or end()
    /// if there is no such DetSet.
    iterator       find(det_id_type id);
    const_iterator find(det_id_type id) const;

    /// Return a reference to the DetSet with the given detector
    /// ID. If there is no such DetSet, we throw an edm::Exception.
    /// **DO NOT MODIFY THE id DATA MEMBER OF THE REFERENCED DetSet!**
    reference       operator[](det_id_type  i);
    const_reference operator[](det_id_type i) const;

    /// Return an iterator to the first DetSet.
    iterator       begin();
    const_iterator begin() const;

    /// Return the off-the-end iterator.
    iterator       end();
    const_iterator end() const;

    /// This function will be called by the edm::Event after the
    /// DetSetVector has been inserted into the Event.
    void post_insert();

  private:
    collection_type   _sets;

    /// Sort the DetSet in order of increasing DetId.
    void _sort();

  };

  template <class T>
  inline
  void
  DetSetVector<T>::swap(DetSetVector<T>& other) 
  {
    _sets.swap(other._sets);
  }

  template <class T>
  inline
  void
  DetSetVector<T>::insert(detset const& t) 
  {
    // It seems we have to sort on each insertion, because we may
    // perform lookups during construction.
    _sets.push_back(t);

    _sort();
  }

  template <class T>
  inline
  typename DetSetVector<T>::reference
  DetSetVector<T>::find_or_insert(det_id_type id)
  {
    std::pair<iterator,iterator> p = 
      std::equal_range(_sets.begin(), _sets.end(), id);

    // If the range isn't empty, we already have the right thing;
    // return a reference to it...
    if ( p.first != p.second ) return *p.first;

    // Insert the right thing, in the right place, and return a
    // reference to the newly inserted thing.
    return *(_sets.insert(p.first, detset(id)));
  }

  template <class T>
  inline
  bool
  DetSetVector<T>::empty() const 
  {
    return _sets.empty();
  }

  template <class T>
  inline
  typename DetSetVector<T>::size_type
  DetSetVector<T>::size() const
  {
    return _sets.size();
  }

  template <class T>
  inline
  typename DetSetVector<T>::iterator
  DetSetVector<T>::find(det_id_type id)
  {
    std::pair<iterator,iterator> p = 
      std::equal_range(_sets.begin(), _sets.end(), id);
    if ( p.first == p.second ) return _sets.end();

    // The range indicated by [p.first, p.second) should be exactly of
    // length 1. It seems likely we don't want to take the time hit of
    // checking this, but here is the appropriate test... We can turn
    // it on if we need the debugging aid.
    #if 0
    assert(std::distance(p.first, p.second) == 1 );
    #endif

    return p.first;
  }

  template <class T>
  inline
  typename DetSetVector<T>::const_iterator
  DetSetVector<T>::find(det_id_type id) const
  {
    std::pair<const_iterator,const_iterator> p = 
      std::equal_range(_sets.begin(), _sets.end(), id);
    if ( p.first == p.second ) return _sets.end();
    // The range indicated by [p.first, p.second) should be exactly of
    // length 1.
    assert( std::distance(p.first, p.second) == 1 );
    return p.first;
  }

  template <class T>
  inline
  typename DetSetVector<T>::reference
  DetSetVector<T>::operator[](det_id_type i) 
  {
    // Find the right DetSet, and return a reference to it.  Throw if
    // there is none.
    iterator it = this->find(i);
    if ( it == this->end() ) detail::_throw_range(i);
    return *it;
  }

  template <class T>
  inline
  typename DetSetVector<T>::const_reference
  DetSetVector<T>::operator[](det_id_type i) const 
  {
    // Find the right DetSet, and return a reference to it.  Throw if
    // there is none.
    const_iterator it = this->find(i);
    if ( it == this->end() ) detail::_throw_range(i);
    return *it;
  }

  template <class T>
  inline
  typename DetSetVector<T>::iterator
  DetSetVector<T>::begin()
  {
    return _sets.begin();
  }

  template <class T>
  inline
  typename DetSetVector<T>::const_iterator
  DetSetVector<T>::begin() const
  {
    return _sets.begin();
  }

  template <class T>
  inline
  typename DetSetVector<T>::iterator
  DetSetVector<T>::end()
  {
    return _sets.end();
  }

  template <class T>
  inline
  typename DetSetVector<T>::const_iterator
  DetSetVector<T>::end() const
  {
    return _sets.end();
  }

  template <class T>
  inline
  void
  DetSetVector<T>::post_insert()
  {
    typename collection_type::iterator i = _sets.begin();
    typename collection_type::iterator e = _sets.end();
    // For each DetSet...
    for ( ; i != e; ++i )
      {
	// sort the Detset pointed to by
	std::sort(i->data.begin(), i->data.end());
      }
  }

  template <class T>
  inline
  void
  DetSetVector<T>::_sort()
  {
    std::sort(_sets.begin(), _sets.end());
  }

}

#endif
