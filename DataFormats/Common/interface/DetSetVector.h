#ifndef DataFormats_Common_DetSetVector_h
#define DataFormats_Common_DetSetVector_h

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

Since CMSSW 2_0_0_pre4, it is possible to skip the automatic sorting
when creating a DetSetVector<T> from an already sorted vector<DetSet<T>>.
If the DSV is not modified afterwards, it will no longer be sorted when
it is inserted in the event.
ONE NOTE OF CAUTION: it is not sufficient to to say that the vector is 
sorted already.  In addition the sorting must have been done with the same
criteria and obey the rules of "strict weak ordering" as will be used to
find things in the collection.  Not insuring this leads to undefined
behavior (usually a core dump).

----------------------------------------------------------------------*/

#include <algorithm>
#include <iterator>
#include <vector>

#include "boost/concept_check.hpp"
#include "boost/mpl/if.hpp"
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#else
#include "boost/bind.hpp"
#endif

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/FillView.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/traits.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/Common/interface/BoolCache.h"

namespace edm {
  class ProductID;

  //------------------------------------------------------------
  // Forward declarations
  template <class T> class DetSetVector;

  //------------------------------------------------------------
  // Helper function, to regularize throwing of exceptions.
  //------------------------------------------------------------

  namespace detail {
    // Throw an edm::Exception with an appropriate message
    inline
    void _throw_range(det_id_type i) {
      Exception::throwThis(errors::InvalidReference,
        "DetSetVector::operator[] called with index not in collection;\nindex value: ", i);
    }
  }

  //------------------------------------------------------------
  //

  // If DetSetVector<T> is instantiated with a class T which inherits
  // from DoNotSortUponInsertion, the resulting class inherits from
  // DoNotSortUponInsertion. In the normal case, DetSetVector<T>
  // inherits from Other.  (This is necessary to assure that
  // DetSetVector<T> is not sorted upon insertion into the Event when
  // T is defined to inherit from DoNotSortUponInsertion).

  template <class T>
  class DetSetVector : 
    public boost::mpl::if_c<boost::is_base_of<edm::DoNotSortUponInsertion, T>::value,
			    edm::DoNotSortUponInsertion,
			    Other>::type
  {
    /// DetSetVector requires that T objects can be compared with
    /// operator<.
    BOOST_CLASS_REQUIRE(T, boost, LessThanComparableConcept);
  public:

    typedef DetSet<T>           detset;
    typedef detset              value_type;
    typedef std::vector<detset> collection_type;

    typedef detset&        reference;
    typedef detset const&  const_reference;

    typedef typename collection_type::iterator       iterator;
    typedef typename collection_type::const_iterator const_iterator;
    typedef typename collection_type::size_type      size_type;

    /// Compiler-generated copy c'tor, d'tor and
    /// assignment are correct.

    /// Create an empty DetSetVector
    DetSetVector();

    /// Create a DetSetVector by copying swapping in the given vector,
    /// and then sorting the contents.
    /// N.B.: Swapping in the vector *destructively modifies the input*.
    /// Using swap here allows us to avoid copying the data.
    /// N.B. 2: if you set alreadySorted to true, data *must* be sorted, 
    /// (the vector<DetSet<T>> must be ordered by detid, and each DetSet
    /// must be ordered according to the natural "strict weak ordering" of Ts.
    /// You *must not* modify the contents after this DSV after creation,
    /// or you might get an undefined behavior / a core dump.
    /// (there are some checks to assure alreadySorted is resetted if you try
    /// to modify the DSV, but you should not count on them)
    explicit DetSetVector(std::vector<DetSet<T> > & input, bool alreadySorted=false);


    void swap(DetSetVector& other);

    DetSetVector& operator= (DetSetVector const& other);

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

   // reserve...
   void reserve(size_t s) { _sets.reserve(s);}

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

    /// Push all the id for each DetSet stored in this DetSetVector
    /// into the given vector 'result'.
    void getIds(std::vector<det_id_type> & result) const;

    /// This function will be called by the edm::Event after the
    /// DetSetVector has been inserted into the Event.
    void post_insert();

    void fillView(ProductID const& id,
		  std::vector<void const*>& pointers,
		  helper_vector& helpers) const;

    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:
    collection_type   _sets;
    edm::BoolCache    _alreadySorted; 

    /// Sort the DetSet in order of increasing DetId.
    void _sort();

  };

  template <class T>
  inline
  DetSetVector<T>::DetSetVector() :
    _sets()
  { }

  template <class T>
  inline
  DetSetVector<T>::DetSetVector(std::vector<DetSet<T> > & input, bool alreadySorted) :
    _sets(), _alreadySorted(alreadySorted)
  {
    _sets.swap(input);
    if (!alreadySorted) _sort();
  }

  template <class T>
  inline
  void
  DetSetVector<T>::swap(DetSetVector<T>& other) {
    _sets.swap(other._sets);
    bool tmp = _alreadySorted; _alreadySorted = other._alreadySorted; other._alreadySorted = tmp;
  }

  template <class T>
  inline
  DetSetVector<T>&
  DetSetVector<T>::operator= (DetSetVector<T> const& other)
  {
    DetSetVector<T> temp(other);
    swap(temp);
    return *this;
  }

  template <class T>
  inline
  void
  DetSetVector<T>::insert(detset const& t) {
    _alreadySorted = false; // we don't know if the DetSet we're adding is already sorted
    // Implementation provided by the Performance Task Force.
    _sets.insert(std::lower_bound(_sets.begin(),
				  _sets.end(),
				  t),
		 t);
#if 0
    // It seems we have to sort on each insertion, because we may
    // perform lookups during construction.
    _sets.push_back(t);

    _sort();
#endif
  }

  template <class T>
  inline
  typename DetSetVector<T>::reference
  DetSetVector<T>::find_or_insert(det_id_type id) {
    // NOTE: we don't have to clear _alreadySorted: the new DS is empty, 
    //       and gets inserted in the correct place
    std::pair<iterator,iterator> p =
      std::equal_range(_sets.begin(), _sets.end(), id);

    // If the range isn't empty, we already have the right thing;
    // return a reference to it...
    if (p.first != p.second) return *p.first;

    // Insert the right thing, in the right place, and return a
    // reference to the newly inserted thing.
#if defined( __GXX_EXPERIMENTAL_CXX0X__)
    return *(_sets.emplace(p.first, id));
#else
    return *(_sets.insert(p.first, detset(id)));
#endif
  }

  template <class T>
  inline
  bool
  DetSetVector<T>::empty() const {
    return _sets.empty();
  }

  template <class T>
  inline
  typename DetSetVector<T>::size_type
  DetSetVector<T>::size() const {
    return _sets.size();
  }

  template <class T>
  inline
  typename DetSetVector<T>::iterator
  DetSetVector<T>::find(det_id_type id) {
    _alreadySorted = false; // it's non const 
    std::pair<iterator,iterator> p =
      std::equal_range(_sets.begin(), _sets.end(), id);
    if (p.first == p.second) return _sets.end();

    // The range indicated by [p.first, p.second) should be exactly of
    // length 1. It seems likely we don't want to take the time hit of
    // checking this, but here is the appropriate test... We can turn
    // it on if we need the debugging aid.
    #if 0
    assert(std::distance(p.first, p.second) == 1);
    #endif

    return p.first;
  }

  template <class T>
  inline
  typename DetSetVector<T>::const_iterator
  DetSetVector<T>::find(det_id_type id) const {
    std::pair<const_iterator,const_iterator> p =
      std::equal_range(_sets.begin(), _sets.end(), id);
    if (p.first == p.second) return _sets.end();
    // The range indicated by [p.first, p.second) should be exactly of
    // length 1.
    assert(std::distance(p.first, p.second) == 1);
    return p.first;
  }

  template <class T>
  inline
  typename DetSetVector<T>::reference
  DetSetVector<T>::operator[](det_id_type i) {
    _alreadySorted = false; // it's non const 
    // Find the right DetSet, and return a reference to it.  Throw if
    // there is none.
    iterator it = this->find(i);
    if (it == this->end()) detail::_throw_range(i);
    return *it;
  }

  template <class T>
  inline
  typename DetSetVector<T>::const_reference
  DetSetVector<T>::operator[](det_id_type i) const {
    // Find the right DetSet, and return a reference to it.  Throw if
    // there is none.
    const_iterator it = this->find(i);
    if (it == this->end()) detail::_throw_range(i);
    return *it;
  }

  template <class T>
  inline
  typename DetSetVector<T>::iterator
  DetSetVector<T>::begin() {
    _alreadySorted = false; // it's non const 
    return _sets.begin();
  }

  template <class T>
  inline
  typename DetSetVector<T>::const_iterator
  DetSetVector<T>::begin() const {
    return _sets.begin();
  }

  template <class T>
  inline
  typename DetSetVector<T>::iterator
  DetSetVector<T>::end() {
    _alreadySorted = false; // it's non const 
    return _sets.end();
  }

  template <class T>
  inline
  typename DetSetVector<T>::const_iterator
  DetSetVector<T>::end() const {
    return _sets.end();
  }


  template <class T>
  inline
  void
  DetSetVector<T>::getIds(std::vector<det_id_type> & result) const
  {
    std::transform(this->begin(), this->end(),
		   std::back_inserter(result),
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
		   std::bind(&DetSet<T>::id,std::placeholders::_1));
#else
		   boost::bind(&DetSet<T>::id,_1));
#endif
  }

  template <class T>
  inline
  void
  DetSetVector<T>::post_insert() {
#ifndef CMS_NOCXX11
    _sets.shrink_to_fit();
#endif
    if (_alreadySorted) return; 
    typename collection_type::iterator i = _sets.begin();
    typename collection_type::iterator e = _sets.end();
    // For each DetSet...
    for (; i != e; ++i) {
#ifndef CMS_NOCXX11
      i->data.shrink_to_fit();
#endif
      // sort the Detset pointed to by
      std::sort(i->data.begin(), i->data.end());
    }
  }

  template <class T>
  inline
  void
  DetSetVector<T>::_sort() {
    std::sort(_sets.begin(), _sets.end());
  }

  template<class T>
  void DetSetVector<T>::fillView(ProductID const& id,
				 std::vector<void const*>& pointers,
				 helper_vector& helpers) const
  {
    detail::reallyFillView(*this, id, pointers, helpers);
  }

  //----------------------------------------------------------------------
  //
  // Free function template to support creation of Views.

  template <class T>
  inline
  void
  fillView(DetSetVector<T> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& pointers,
	   helper_vector& helpers)
  {
    obj.fillView(id, pointers, helpers);
  }

  template <class T>
  struct has_fillView<edm::DetSetVector<T> >
  {
    static bool const value = true;
  };


  // Free swap function
  template <class T>
  inline
  void
  swap(DetSetVector<T>& a, DetSetVector<T>& b) 
  {
    a.swap(b);
  }

}


//specialize behavior of edm::Ref to get access to the 'Det'
namespace edm {

  namespace refhelper {
    template<typename T>
    class FindForDetSetVector : public std::binary_function<const DetSetVector<T>&, std::pair<det_id_type, typename DetSet<T>::collection_type::size_type>, const T*> {
    public:
      typedef FindForDetSetVector<T> self;
      typename self::result_type operator()(typename self::first_argument_type iContainer, typename self::second_argument_type iIndex) {
        return &(*(iContainer.find(iIndex.first)->data.begin()+iIndex.second));
      }
    };

    template<typename T>
      struct FindTrait<DetSetVector<T>,T> {
        typedef FindForDetSetVector<T> value;
      };
  }

   //helper function to make it easier to create a edm::Ref

  template<class HandleT>
  Ref<typename HandleT::element_type, typename HandleT::element_type::value_type::value_type>
  makeRefTo(const HandleT& iHandle,
             det_id_type iDetID,
             typename HandleT::element_type::value_type::const_iterator itIter) {
    typedef typename HandleT::element_type Vec;
    typename Vec::value_type::collection_type::size_type index=0;
    typename Vec::const_iterator itFound = iHandle->find(iDetID);
    if(itFound == iHandle->end()) {
      Exception::throwThis(errors::InvalidReference,
        "an edm::Ref to an edm::DetSetVector was given a DetId, ", iDetID, ", that is not in the DetSetVector");
    }
    index += (itIter- itFound->data.begin());
    if(index >= itFound->data.size()) {
      Exception::throwThis(errors::InvalidReference,
        "an edm::Ref to a edm::DetSetVector is being made with an interator that is not part of the edm::DetSet itself");
    }
    return Ref<typename HandleT::element_type,
	       typename HandleT::element_type::value_type::value_type>
	      (iHandle,std::make_pair(iDetID,index));
  }

  template<class HandleT>
  Ref<typename HandleT::element_type, typename HandleT::element_type::value_type::value_type>
  makeRefToDetSetVector(const HandleT& iHandle,
             det_id_type iDetID,
             typename HandleT::element_type::value_type::iterator itIter) {
    typedef typename HandleT::element_type Vec;
    typename Vec::detset::const_iterator itIter2 = itIter;
    return makeRefTo(iHandle,iDetID,itIter2);
  }
}
#endif
