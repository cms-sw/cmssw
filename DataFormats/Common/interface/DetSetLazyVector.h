#ifndef DataFormats_Common_DetSetLazyVector_h
#define DataFormats_Common_DetSetLazyVector_h

/*----------------------------------------------------------------------
  
DetSetLazyVector: A collection of homogeneous objects that can be used for
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
#include "boost/iterator/transform_iterator.hpp"
#include "boost/shared_ptr.hpp"

#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/Ref.h"

namespace edm {

  //------------------------------------------------------------
  // Forward declarations
  template <class T> class DetSetLazyVector;

  //------------------------------------------------------------
  // Helper function, to regularize throwing of exceptions.
  //------------------------------------------------------------

  namespace dslvdetail
  {
    // Throw an edm::Exception with an appropriate message
    inline
    void _throw_range(det_id_type i)
  {
      Exception::throwThis(errors::InvalidReference,
        "DetSetLazyVector::operator[] called with index not in collection;\nindex value: ", i);
  }
  }
  
  namespace dslv {
    template< typename T>
    class LazyGetter {
public:
      virtual ~LazyGetter() {}
      virtual void fill(DetSet<T>&) = 0;
    };
    template<typename T>
      struct LazyAdapter : public std::unary_function<const DetSet<T>&, const DetSet<T>&> {
        LazyAdapter(boost::shared_ptr<LazyGetter<T> > iGetter): getter_(iGetter) {}
        const DetSet<T>& operator()(const DetSet<T>& iUpdate) const {
          if(iUpdate.data.empty() && getter_) {
            //NOTE: because this is 'updating a cache' we need to cast away the const
	    DetSet<T>& temp = const_cast< DetSet<T>& >(iUpdate);
            getter_->fill(temp);
	    std::sort(temp.begin(),temp.end());
          }
          return iUpdate;
        }
private:
        boost::shared_ptr<LazyGetter<T> > getter_;
      };
  }
  //------------------------------------------------------------
  //
  namespace refhelper {
    template<typename T>
    struct FindDetSetForDetSetLazyVector : public std::binary_function<const DetSetLazyVector<T>&, edm::det_id_type, const DetSet<T>*> {
      typedef FindDetSetForDetSetLazyVector<T> self;
      typename self::result_type operator()(typename self::first_argument_type iContainer,  typename self::second_argument_type iIndex) const {
        return &(*(iContainer.find(iIndex)));
      }
    };
  }
  
  template <class T>
  class DetSetLazyVector 
  {
    /// DetSetVector requires that T objects can be compared with
    /// operator<.
    BOOST_CLASS_REQUIRE(T, boost, LessThanComparableConcept);
  public:

    typedef DetSet<T>           detset;
    typedef detset              value_type;
    typedef std::vector<value_type> collection_type;

    typedef detset const&  const_reference;

    //iterator returns a DetSet<T> instead of a Ref<...>
    typedef boost::transform_iterator< dslv::LazyAdapter<T>, typename collection_type::const_iterator > const_iterator;
    typedef typename collection_type::size_type      size_type;

    /// Compiler-generated default c'tor, copy c'tor, d'tor and
    /// assignment are correct.

    // Add the following only if needed.
    //template <class InputIterator>
    //DetSetLazyVector(InputIterator b, InputIterator e);

    DetSetLazyVector() {}
    
    DetSetLazyVector(boost::shared_ptr<dslv::LazyGetter<T> > iGetter, const std::vector<det_id_type>& iDets) :
    sets_(),
    getter_(iGetter) {
        sets_.reserve(iDets.size());
        det_id_type sanityCheck = 0;
        for(std::vector<det_id_type>::const_iterator itDetId = iDets.begin(), itDetIdEnd = iDets.end();
            itDetId != itDetIdEnd;
            ++itDetId) {
          assert(sanityCheck <= *itDetId && "vector of det_id_type was not ordered");
          sanityCheck = *itDetId;
          sets_.push_back(DetSet<T>(*itDetId));
        }
      }

    void swap(DetSetLazyVector& other);

    //    DetSetVector& operator=(DetSetVector const& rhs);

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
    boost::shared_ptr<dslv::LazyGetter<T> > getter_;
  };

  template <class T>
  inline
  void
  DetSetLazyVector<T>::swap(DetSetLazyVector<T>& other) 
  {
    sets_.swap(other.sets_);
    std::swap(getter_,other.getter_);
  }

  template <class T>
  inline
  bool
  DetSetLazyVector<T>::empty() const 
  {
    return sets_.empty();
  }

  template <class T>
  inline
  typename DetSetLazyVector<T>::size_type
  DetSetLazyVector<T>::size() const
  {
    return sets_.size();
  }

  template <class T>
  inline
  typename DetSetLazyVector<T>::const_iterator
  DetSetLazyVector<T>::find(det_id_type id) const
  {
    if(empty()) {
      dslv::LazyAdapter<T> adapter(getter_);
      return boost::make_transform_iterator(sets_.end(),adapter);
    }
    //NOTE: using collection_type::const_iterator and NOT const_iterator. We do this to avoid calling the
    // dereferencing operation on const_iterator which would cause the 'lazy update' to happen
    std::pair<typename collection_type::const_iterator,typename collection_type::const_iterator> p = 
    std::equal_range(sets_.begin(), sets_.end(), id);
    if (p.first == p.second) {
      dslv::LazyAdapter<T> adapter(getter_);
      return boost::make_transform_iterator(sets_.end(),adapter);
    }
    // The range indicated by [p.first, p.second) should be exactly of
    // length 1.
    assert(std::distance(p.first, p.second) == 1);
    dslv::LazyAdapter<T> adapter(getter_);
    return boost::make_transform_iterator(p.first,adapter);
  }

  template <class T>
  inline
  typename DetSetLazyVector<T>::const_reference
  DetSetLazyVector<T>::operator[](det_id_type i) const 
  {
    // Find the right DetSet, and return a reference to it.  Throw if
    // there is none.
    const_iterator it = this->find(i);
    if (it == this->end()) dslvdetail::_throw_range(i);
    return *it;
  }

  template <class T>
  inline
  typename DetSetLazyVector<T>::const_iterator
  DetSetLazyVector<T>::begin() const
  {
    dslv::LazyAdapter<T> adapter(getter_);
    return boost::make_transform_iterator(sets_.begin(),adapter);
  }

  template <class T>
  inline
  typename DetSetLazyVector<T>::const_iterator
  DetSetLazyVector<T>::end() const
  {
    dslv::LazyAdapter<T> adapter(getter_);
    return boost::make_transform_iterator(sets_.end(),adapter);
  }

  // Free swap function
  template <class T>
  inline
  void
  swap(DetSetLazyVector<T>& a, DetSetLazyVector<T>& b) 
  {
    a.swap(b);
  }

//specialize behavior of edm::Ref to get access to the 'Det'

  namespace refhelper {
    template<typename T>
    struct FindForDetSetLazyVector : public std::binary_function<const DetSetLazyVector<T>&, std::pair<det_id_type, typename DetSet<T>::collection_type::size_type>, const T*> {
      typedef FindForDetSetLazyVector<T> self;
      typename self::result_type operator()(typename self::first_argument_type iContainer,  typename self::second_argument_type iIndex) {
        return &(*(iContainer.find(iIndex.first)->data.begin()+iIndex.second));
      }
    };
    
    template<typename T> 
      struct FindTrait<DetSetLazyVector<T>,T> {
        typedef FindForDetSetLazyVector<T> value;
      };
  }
  
   //helper function to make it easier to create a edm::Ref
   
   template<class HandleT>
   Ref< typename HandleT::element_type, typename HandleT::element_type::value_type::value_type> 
   makeRefToDetSetLazyVector(const HandleT& iHandle, 
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
   makeRefToDetSetLazyVector(const HandleT& iHandle, 
             det_id_type iDetID,
             typename HandleT::element_type::value_type::iterator itIter) {
      typedef typename HandleT::element_type Vec;
      typename Vec::detset::const_iterator itIter2 = itIter;
      return makeRefToDetSetLazyVector(iHandle,iDetID,itIter2);
   }
}  
#endif
