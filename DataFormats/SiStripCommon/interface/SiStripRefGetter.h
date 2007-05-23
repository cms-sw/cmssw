#ifndef DataFormats_SiStripCommon_SiStripRefGetter_h
#define DataFormats_SiStripCommon_SiStripRefGetter_h

#include <algorithm>
#include <vector>

#include "boost/concept_check.hpp"
#include "boost/iterator/indirect_iterator.hpp"

#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/SiStripCommon/interface/SiStripLazyGetter.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

  //------------------------------------------------------------

  template <class T, class C> class SiStripRefGetter;

  //------------------------------------------------------------
  // Helper function, to regularize throwing of exceptions.
  //------------------------------------------------------------

  namespace rgdetail
  {
    // Throw an edm::Exception with an appropriate message
    inline void _throw_range(uint32_t region)
    {
      throw Exception(errors::InvalidReference)
	<< "SiStripRefGetter::operator[] called with index not in collection;\n"
	<< "index value: " << region;
    }
  }

  //------------------------------------------------------------
  //
  
  template<typename T, typename C >
    struct FindRegion : public std::binary_function<const C &, uint32_t, const RegionIndex<T>*> {
      typedef FindRegion<T,C> self;
      typename self::result_type operator()(typename self::first_argument_type iContainer,  typename self::second_argument_type iIndex) const {
        return &(*(iContainer.begin()+iIndex));
      }
    };
  
  template<typename T, typename C >
    struct FindValue : public std::binary_function<const C &, typename C::record_type::const_iterator, const T* > {
      typedef FindValue<T,C> self;
      typename self::result_type operator()(typename self::first_argument_type container, typename self::second_argument_type iter) const {
        return &(*iter);
      }
    };
  
  template <class T, class C=SiStripLazyGetter<T> >
  class SiStripRefGetter 
  {

    BOOST_CLASS_REQUIRE(T, boost, LessThanComparableConcept);

  public:

    typedef Ref< SiStripRefGetter<T,C>, T, FindValue< T,SiStripRefGetter<T,C> > > self_ref;
    typedef Ref< C, typename C::value_type, FindRegion<T,C> > ref_type;
    typedef std::vector<ref_type> collection_type;
    typedef typename C::record_type record_type;
    typedef typename C::const_reference  const_reference;
    typedef boost::indirect_iterator<typename collection_type::const_iterator> const_iterator;
    typedef typename collection_type::size_type size_type;

    SiStripRefGetter() {}
    
    template <typename THandle>
      SiStripRefGetter(const THandle& iHandle, const std::vector<uint32_t>& iRegions) : sets_() {
        sets_.reserve(iRegions.size());
        for (std::vector<uint32_t>::const_iterator iRegion = iRegions.begin();
	    iRegion != iRegions.end();
            ++iRegion) {
          //the last 'false' says to not get the data right now
          sets_.push_back(ref_type(iHandle, *iRegion, false));
        }
      }

    void swap(SiStripRefGetter& other);

    /// Return true if we contain no 'ref_type's (one per Region).
    bool empty() const;

    /// Return the number of contained 'ref_type's (one per Region).
    size_type size() const;

    /// Return an iterator to the 'ref_type' for a given Region id, or end() 
    /// if there is no such Region.
    const_iterator find(uint32_t region) const;

    /// Return a reference to the 'ref_type' for a given Region id, or throw 
    /// an edm::Exception if there is no such Region.
    const_reference operator[](uint32_t region) const;

    /// Return an iterator to the first 'ref_type'.
    const_iterator begin() const;

    /// Return the off-the-end iterator.
    const_iterator end() const;

  private:
    collection_type   sets_;

  };

  template <class T, class C>
  inline
  void
  SiStripRefGetter<T,C>::swap(SiStripRefGetter<T,C>& other) 
  {
    sets_.swap(other.sets_);
  }

  template <class T, class C>
  inline
  bool
  SiStripRefGetter<T,C>::empty() const 
  {
    return sets_.empty();
  }

  template <class T, class C>
  inline
  typename SiStripRefGetter<T,C>::size_type
  SiStripRefGetter<T,C>::size() const
  {
    return sets_.size();
  }

  template <class T, class C>
  inline
  typename SiStripRefGetter<T,C>::const_iterator
  SiStripRefGetter<T,C>::find(uint32_t region) const
  {
    if (size() < region+1) return sets_.end();
    typename collection_type::const_iterator it = sets_.begin()+region;
    return it;
  }

  template <class T, class C>
  inline
  typename SiStripRefGetter<T,C>::const_reference
  SiStripRefGetter<T,C>::operator[](uint32_t region) const 
  {
    const_iterator it = this->find(region);
    if (it == this->end()) rgdetail::_throw_range(region);
    return *it;
  }

  template <class T, class C>
  inline
  typename SiStripRefGetter<T,C>::const_iterator
  SiStripRefGetter<T,C>::begin() const
  {
    return sets_.begin();
  }

  template <class T, class C>
  inline
  typename SiStripRefGetter<T,C>::const_iterator
  SiStripRefGetter<T,C>::end() const
  {
    return sets_.end();
  }

  template <class T, class C>
  inline
  void
  swap(SiStripRefGetter<T,C>& a, SiStripRefGetter<T,C>& b) 
  {
    a.swap(b);
  }

 //helper function to make it easier to create a edm::Ref

  template<class HandleT>
    typename HandleT::element_type::self_ref
    makeRefToSiStripRefGetter(const HandleT& iHandle, typename HandleT::element_type::record_type::const_iterator iter) {
    return typename HandleT::element_type::self_ref(iHandle,iter,false);
  }

#if ! GCC_PREREQUISITE(3,4,4)
  // Has swap function
  template <class T, class C>
  struct has_swap<edm::SiStripRefGetter<T,C> > {
    static bool const value = true;
  };

#endif

}
  
#endif

