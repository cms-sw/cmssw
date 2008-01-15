#ifndef DataFormats_SiStripCommon_SiStripRefGetter_H
#define DataFormats_SiStripCommon_SiStripRefGetter_H

#include <algorithm>
#include <vector>
#include <iostream>
#include <utility>
#include "boost/concept_check.hpp"
#include "boost/iterator/indirect_iterator.hpp"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripLazyGetter.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

  template<class T> class SiStripRefGetter {

    BOOST_CLASS_REQUIRE(T, boost, LessThanComparableConcept);

  public:

    typedef Ref< SiStripLazyGetter<T>, RegionIndex<T>, FindRegion<T> > region_ref;
    typedef std::vector<region_ref> collection_type;
    typedef boost::indirect_iterator<typename collection_type::const_iterator> const_iterator;
    typedef std::vector<T> record_type;
    typedef std::pair<typename record_type::const_iterator, typename record_type::const_iterator> record_pair;

    /// Default constructor. Default maximum region number 50,000.
    SiStripRefGetter(uint32_t=50000);
    
    /// Constructor with regions.
    SiStripRefGetter(const edm::Handle< SiStripLazyGetter<T> >&, const std::vector<uint32_t>&);

    /// Reserve memory for sets_ collection.
    void reserve(uint32_t);

    /// Swap contents of class.
    void swap(SiStripRefGetter& other);

    /// Add a new region to the end of the collection.
    void push_back(const edm::Handle<SiStripLazyGetter<T> >&, const uint32_t&);

    /// Return true if we contain no 'region_ref's (one per Region).
    bool empty() const;

    /// Return the number of contained 'region_ref's (one per Region).
    uint32_t size() const;

    /// Return a reference to the RegionIndex<T> for a given region.
    const RegionIndex<T>& operator[](uint32_t) const;

    /// Returns a reference to the last RegionIndex<T> added to the 
    /// collection, or throws an exception if empty.
    const RegionIndex<T>& back() const;

    /// Return an iterator to the first RegionIndex<T>.
    const_iterator begin() const;

    /// Return the off-the-end iterator.
    const_iterator end() const;

    /// Returns true if region already defined. 
    bool find(uint32_t) const;

  private:

    collection_type sets_;
    std::vector<uint32_t> regions_;
  };
  
  template <class T>
    inline
    SiStripRefGetter<T>::SiStripRefGetter(uint32_t maxindex) : sets_(), regions_(maxindex/32+1,0)
    {}

  template <class T>
    inline
    SiStripRefGetter<T>::SiStripRefGetter(const edm::Handle< SiStripLazyGetter<T> >& getter, const std::vector<uint32_t>& interest) : sets_(), regions_(getter->regions()/32+1,0) 
    {
      sets_.reserve(interest.size());
      for (uint32_t index=0;index<interest.size();index++) {
	sets_.push_back(region_ref(getter,interest[index],false));
      }
    }
  
  template <class T>
    inline
    void
    SiStripRefGetter<T>::reserve(uint32_t size) 
    {
      sets_.reserve(size);
    }
  
  template <class T>
    inline
    void
    SiStripRefGetter<T>::swap(SiStripRefGetter<T>& other) 
    {
      sets_.swap(other.sets_);
      regions_.swap(other.regions_);
    }
  
  template <class T>
    inline
    void 
    SiStripRefGetter<T>::push_back(const edm::Handle< SiStripLazyGetter<T> >& getter, const uint32_t& index)
    {
      sets_.push_back(region_ref(getter, index, false));
      regions_[index/32] = regions_[index/32]|(1<<index%32);
    }
  
  template <class T>
    inline
    bool
    SiStripRefGetter<T>::empty() const 
    {
      return sets_.empty();
    }

  template <class T>
    inline
    uint32_t
    SiStripRefGetter<T>::size() const
    {
      return sets_.size();
    }

  template <class T>
    inline
    const RegionIndex<T>&
    SiStripRefGetter<T>::operator[](uint32_t index) const 
    {
      if (size() < index+1) sistripdetail::_throw_range(index);
      const_iterator it = sets_.begin()+index;
      return *it;
    }

 template <class T>
    inline
    const RegionIndex<T>&
    SiStripRefGetter<T>::back() const 
    {
      if (empty()) sistripdetail::_throw_range(0);
      return (*this)[size()-1];
    } 
  
  template <class T>
    inline
    typename SiStripRefGetter<T>::const_iterator
    SiStripRefGetter<T>::begin() const
    {
      return sets_.begin();
    }
  
  template <class T>
    inline
    typename SiStripRefGetter<T>::const_iterator
    SiStripRefGetter<T>::end() const
    {
      return sets_.end();
    }
  
  template <class T>
    inline
    bool 
    SiStripRefGetter<T>::find(uint32_t index) const
    {
      return (regions_[index/32]>>(index%32))&1;
    }

  template <class T>
    inline
    void
    swap(SiStripRefGetter<T>& a, SiStripRefGetter<T>& b) 
    {
      a.swap(b);
    }
  
  //------------------------------------------------------------
  
#if ! GCC_PREREQUISITE(3,4,4)
  // Has swap function
  template <class T>
  struct has_swap<edm::SiStripRefGetter<T> > {
    static bool const value = true;
  };

#endif

}
  
#endif

