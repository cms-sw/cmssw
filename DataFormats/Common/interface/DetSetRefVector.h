#ifndef DataFormats_Common_DetSetRefVector_h
#define DataFormats_Common_DetSetRefVector_h

/*----------------------------------------------------------------------
  
DetSeReftVector: A collection of homogeneous objects that can be used for
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

----------------------------------------------------------------------*/

#include <algorithm>
#include <vector>

#include "boost/concept_check.hpp"
#include "boost/iterator/indirect_iterator.hpp"

#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/TestHandle.h"

namespace edm {

  //------------------------------------------------------------
  // Forward declarations
  template <typename T, typename C> class DetSetRefVector;

  //------------------------------------------------------------
  // Helper function, to regularize throwing of exceptions.
  //------------------------------------------------------------

  namespace dsrvdetail
  {
    // Throw an edm::Exception with an appropriate message
    inline
    void _throw_range(det_id_type i)
    {
      Exception::throwThis(errors::InvalidReference,
        "DetSetRefVector::operator[] called with index not in collection;\nindex value: ", i);
    }
  }

  //------------------------------------------------------------
  //
  namespace refhelper {
    template<typename T, typename C >
    struct FindDetSetForDetSetVector : public std::binary_function<const C &, edm::det_id_type, const DetSet<T>*> {
      typedef FindDetSetForDetSetVector<T,C> self;
      typename self::result_type operator()(typename self::first_argument_type iContainer,  typename self::second_argument_type iIndex) const {
        return &(*(iContainer.find(iIndex)));
      }
    };
  }
  
  //allow comparison of edm::Ref<...> to the det_it_type.  This allows searching without dereferencing the edm::Ref
  template <typename T, typename C=DetSetVector<T> >
    struct CompareRefDetSet {
    typedef Ref<C, DetSet<T>, refhelper::FindDetSetForDetSetVector<T,C> > ref_type; 
      bool operator()(const ref_type& iRef, det_id_type iId) {
        return iRef.key() < iId;
      }
      bool operator()(det_id_type iId, const ref_type& iRef) {
        return iId < iRef.key();
      }
    };

  template <typename T, typename C=DetSetVector<T> >
  class DetSetRefVector 
  {
    /// DetSetVector requires that T objects can be compared with
    /// operator<.
    BOOST_CLASS_REQUIRE(T, boost, LessThanComparableConcept);
  public:

    typedef DetSet<T>           detset;
    typedef detset              value_type;
    typedef Ref<C, DetSet<T>, refhelper::FindDetSetForDetSetVector<T,C> > ref_type;
    typedef std::vector<ref_type> collection_type;

    typedef detset const&  const_reference;

    //iterator returns a DetSet<T> instead of a Ref<...>
    typedef boost::indirect_iterator<typename collection_type::const_iterator> const_iterator;
    typedef typename collection_type::size_type      size_type;

    /// Compiler-generated default c'tor, copy c'tor, d'tor and
    /// assignment are correct.

    // Add the following only if needed.
    //template <class InputIterator>
    //DetSetRefVector(InputIterator b, InputIterator e);

    DetSetRefVector() {}
    
      DetSetRefVector(const Handle<C>& iHandle, const std::vector<det_id_type>& iDets) : sets_() {
        sets_.reserve(iDets.size());
        det_id_type sanityCheck = 0;
        for(std::vector<det_id_type>::const_iterator itDetId = iDets.begin(),
            itDetIdEnd = iDets.end();
            itDetId != itDetIdEnd;
            ++itDetId) {
          assert(sanityCheck <= *itDetId && "vector of det_id_type was not ordered");
          sanityCheck = *itDetId;
          //the last 'false' says to not get the data right now
          sets_.push_back(ref_type(iHandle, *itDetId, false));
        }
      }

      DetSetRefVector(const OrphanHandle<C>& iHandle, const std::vector<det_id_type>& iDets) : sets_() {
        sets_.reserve(iDets.size());
        det_id_type sanityCheck = 0;
        for(std::vector<det_id_type>::const_iterator itDetId = iDets.begin(),
            itDetIdEnd = iDets.end();
            itDetId != itDetIdEnd;
            ++itDetId) {
          assert(sanityCheck <= *itDetId && "vector of det_id_type was not ordered");
          sanityCheck = *itDetId;
          //the last 'false' says to not get the data right now
          sets_.push_back(ref_type(iHandle, *itDetId, false));
        }
      }

      DetSetRefVector(const TestHandle<C>& iHandle, const std::vector<det_id_type>& iDets) : sets_() {
        sets_.reserve(iDets.size());
        det_id_type sanityCheck = 0;
        for(std::vector<det_id_type>::const_iterator itDetId = iDets.begin(),
            itDetIdEnd = iDets.end();
            itDetId != itDetIdEnd;
            ++itDetId) {
          assert(sanityCheck <= *itDetId && "vector of det_id_type was not ordered");
          sanityCheck = *itDetId;
          //the last 'false' says to not get the data right now
          sets_.push_back(ref_type(iHandle, *itDetId, false));
        }
      }

    void swap(DetSetRefVector& other);

    DetSetRefVector& operator=(DetSetRefVector const& rhs);

    /// Return true if we contain no DetSets.
    bool empty() const;

    /// Return the number of contained DetSets
    size_type size() const;

    // Do we need a short-hand method to return the number of T
    // instances? If so, do we optimize for size (calculate on the
    // fly) or speed (keep a current cache)?

    /// Return an iterator to the DetSet with the given id, or end()
    /// if there is no such DetSet.
    const_iterator find(det_id_type id) const;

    /// Return a reference to the DetSet with the given detector
    /// ID. If there is no such DetSet, we throw an edm::Exception.
    /// **DO NOT MODIFY THE id DATA MEMBER OF THE REFERENCED DetSet!**
    const_reference operator[](det_id_type i) const;

    /// Return an iterator to the first DetSet.
    const_iterator begin() const;

    /// Return the off-the-end iterator.
    const_iterator end() const;

    /// This function will be called by the edm::Event after the
    /// DetSetVector has been inserted into the Event.
    //void post_insert();

  private:
    collection_type   sets_;

  };

  template <typename T, typename C>
  inline
  void
  DetSetRefVector<T,C>::swap(DetSetRefVector<T,C>& other) {
    sets_.swap(other.sets_);
  }

  template <typename T, typename C>
  inline
  DetSetRefVector<T ,C>&
  DetSetRefVector<T, C>::operator=(DetSetRefVector<T,C> const& rhs) {
    DetSetRefVector<T, C> temp(rhs);
    this->swap(temp);
    return *this;
  }

  template <typename T, typename C>
  inline
  bool
  DetSetRefVector<T,C>::empty() const 
  {
    return sets_.empty();
  }

  template <typename T, typename C>
  inline
  typename DetSetRefVector<T,C>::size_type
  DetSetRefVector<T,C>::size() const
  {
    return sets_.size();
  }

  template <typename T, typename C>
  inline
  typename DetSetRefVector<T,C>::const_iterator
  DetSetRefVector<T,C>::find(det_id_type id) const
  {
    if(empty()) {
      return sets_.end();
    }
    std::pair<typename collection_type::const_iterator,typename collection_type::const_iterator> p = 
    std::equal_range(sets_.begin(), sets_.end(), id, CompareRefDetSet<T,C>());
    if (p.first == p.second) return sets_.end();
    // The range indicated by [p.first, p.second) should be exactly of
    // length 1.
    assert(std::distance(p.first, p.second) == 1);
    return p.first;
  }

  template <typename T, typename C>
  inline
  typename DetSetRefVector<T,C>::const_reference
  DetSetRefVector<T,C>::operator[](det_id_type i) const 
  {
    // Find the right DetSet, and return a reference to it.  Throw if
    // there is none.
    const_iterator it = this->find(i);
    if (it == this->end()) dsrvdetail::_throw_range(i);
    return *it;
  }

  template <typename T, typename C>
  inline
  typename DetSetRefVector<T,C>::const_iterator
  DetSetRefVector<T,C>::begin() const
  {
    return sets_.begin();
  }

  template <typename T, typename C>
  inline
  typename DetSetRefVector<T,C>::const_iterator
  DetSetRefVector<T,C>::end() const
  {
    return sets_.end();
  }

  // Free swap function
  template <typename T, typename C>
  inline
  void
  swap(DetSetRefVector<T, C>& a, DetSetRefVector<T, C>& b) {
    a.swap(b);
  }

//specialize behavior of edm::Ref to get access to the 'Det'

  namespace refhelper {
    template<typename T, typename C>
    struct FindForDetSetRefVector : public std::binary_function<const DetSetRefVector<T,C>&, std::pair<det_id_type, typename DetSet<T>::collection_type::size_type>, const T*> {
      typedef FindForDetSetRefVector<T,C> self;
      typename self::result_type operator()(typename self::first_argument_type iContainer,  typename self::second_argument_type iIndex) {
        return &(*(iContainer.find(iIndex.first)->data.begin()+iIndex.second));
      }
    };
    
    template<typename T, typename C> 
      struct FindTrait<DetSetRefVector<T,C>,T> {
        typedef FindForDetSetRefVector<T,C> value;
      };
  }
  
   //helper function to make it easier to create a edm::Ref
   
   template<class HandleT>
   Ref< typename HandleT::element_type, typename HandleT::element_type::value_type::value_type> 
   makeRefToDetSetRefVector(const HandleT& iHandle, 
             det_id_type iDetID,
             typename HandleT::element_type::value_type::const_iterator itIter) {
      typedef typename HandleT::element_type Vec;
      typename Vec::value_type::collection_type::size_type index=0;
      typename Vec::const_iterator itFound = iHandle->find(iDetID);
      index += (itIter- itFound->data.begin());
      return    Ref< typename HandleT::element_type, typename HandleT::element_type::value_type::value_type> (iHandle,std::make_pair(iDetID,index));
   }
   
   template<class HandleT>
   Ref< typename HandleT::element_type, typename HandleT::element_type::value_type::value_type> 
   makeRefToDetSetRefVector(const HandleT& iHandle, 
             det_id_type iDetID,
             typename HandleT::element_type::value_type::iterator itIter) {
      typedef typename HandleT::element_type Vec;
      typename Vec::detset::const_iterator itIter2 = itIter;
      return makeRefToDetSetRefVector(iHandle,iDetID,itIter2);
   }
}  
#endif
