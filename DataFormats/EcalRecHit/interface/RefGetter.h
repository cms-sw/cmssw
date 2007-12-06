#ifndef DataFormats_Common_RefGetter_H
#define DataFormats_Common_RefGetter_H

#include <algorithm>
#include <vector>
#include <iostream>
#include <utility>
#include "boost/concept_check.hpp"
#include "boost/iterator/indirect_iterator.hpp"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalRecHit/interface/LazyGetter.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

  //------------------------------------------------------------

  template <class T> class RefGetter;

  template<typename T>
    class FindValue : public std::binary_function< const RefGetter<T>&, typename std::vector<T>::const_iterator, const T* > 
    {
      public :
	typedef FindValue<T> self;
	typename self::result_type operator()
	(typename self::first_argument_type container, typename self::second_argument_type iter) const 
	{
	  return &(*iter);
	}
    };

  //------------------------------------------------------------
  
  class RegionRecord
    {
    public: 
      RegionRecord(uint32_t nregions) : regions_(nregions/32+1,0) {}
      ~RegionRecord() {}
      void record(uint32_t region) {regions_[region/32] = regions_[region/32]|(1<<region%32);}
      bool recorded(uint32_t region) const {return (regions_[region/32]>>(region%32))&1;}
    private:
      RegionRecord();
      std::vector<uint32_t> regions_;
    };
  
  //------------------------------------------------------------

  template <class T>
  class RefGetter 
  {

    BOOST_CLASS_REQUIRE(T, boost, LessThanComparableConcept);

  public:

    typedef std::vector< RegionIndex<T> > register_type;
    typedef std::vector<T> record_type;
    typedef typename record_type::const_iterator record_iterator;
    typedef std::pair<typename record_type::const_iterator, typename record_type::const_iterator> record_pair;
    typedef Ref< RefGetter<T>, T, FindValue<T> > value_ref;
    typedef Ref< LazyGetter<T>, RegionIndex<T>, FindRegion<T> > region_ref;
    typedef std::vector<region_ref> collection_type;
    typedef boost::indirect_iterator<typename collection_type::const_iterator> const_iterator;

    /// Default constructor. Default maximum region number 50,000.
    RefGetter(uint32_t=50000);
    
    /// Constructor with regions and Handle
    RefGetter(const edm::Handle< LazyGetter<T> >&, const std::vector<uint32_t>&);

    /// Reserve memory for sets_ collection.
    void reserve(uint32_t);

    /// Swap contents of class.
    void swap(RefGetter& other);

    /// Add a new region to the end of the collection.
    void push_back(const edm::Handle<LazyGetter<T> >&, const uint32_t&);

    /// Return true if we contain no 'region_ref's (one per Region).
    bool empty() const;

    /// Return the number of contained 'region_ref's (one per Region).
    uint32_t size() const;

    /// Return a reference to the RegionIndex<T> for a given region.
    const RegionIndex<T>& operator[](uint32_t) const;

    /// Returns a reference to the last RegionIndex<T> added to the 
    /// collection, or throws an exception if empty.
    const RegionIndex<T>& back() const;
    
    /// Returns start end end iterators for values of a given det-id 
    /// within the last RegionIndex<T> added to the collection.
    record_pair back(uint32_t) const;

    /// Return an iterator to the first RegionIndex<T>.
    const_iterator begin() const;

    /// Return the off-the-end iterator.
    const_iterator end() const;

    /// Returns true if region already defined. 
    bool find(uint32_t) const;

  private:

    collection_type sets_;
    RegionRecord regions_;
  };
  
  template <class T>
    inline
    RefGetter<T>::RefGetter(uint32_t maxindex) : sets_(), regions_(maxindex)
    {}

    template <class T>
    inline
    RefGetter<T>::RefGetter(const edm::Handle<LazyGetter<T> >& getter, const std::vector<uint32_t>& interest) : sets_(), regions_(getter->regions()) 
    {
      sets_.reserve(interest.size());
      for (uint32_t index=0;index<interest.size();index++) {
	sets_.push_back(region_ref(getter,interest[index],false));
	regions_.record(interest[index]);
      }
    }
  
  template <class T>
    inline
    void
    RefGetter<T>::reserve(uint32_t size) 
    {
      sets_.reserve(size);
    }
  
  template <class T>
    inline
    void
    RefGetter<T>::swap(RefGetter<T>& other) 
    {
      sets_.swap(other.sets_);
    }

   template <class T>
     inline
     void 
     RefGetter<T>::push_back(const edm::Handle< LazyGetter<T> >& getter, const uint32_t& index)
     {
       sets_.push_back(region_ref(getter, index, false));
       regions_.record(index);
     }
  
  template <class T>
    inline
    bool
    RefGetter<T>::empty() const 
    {
      return sets_.empty();
    }
  
  template <class T>
    inline
    uint32_t
    RefGetter<T>::size() const
    {
      return sets_.size();
    }

  template <class T>
    inline
    const RegionIndex<T>&
    RefGetter<T>::operator[](uint32_t index) const 
    {
      if (size() < index+1) edm::lazydetail::_throw_range(index);
      const_iterator it = sets_.begin()+index;
      return *it;
    }

 template <class T>
    inline
    const RegionIndex<T>&
    RefGetter<T>::back() const 
    {
      if (empty()) edm::lazydetail::_throw_range(0);
      return (*this)[size()-1];
    } 

  template <class T>
    inline
    typename RefGetter<T>::record_pair
    RefGetter<T>::back(uint32_t detid) const 
    {
      return back().find(detid);
    } 
  
  template <class T>
    inline
    typename RefGetter<T>::const_iterator
    RefGetter<T>::begin() const
    {
      return sets_.begin();
    }
  
  template <class T>
    inline
    typename RefGetter<T>::const_iterator
    RefGetter<T>::end() const
    {
      return sets_.end();
    }
  
  template <class T>
    inline
    bool 
    RefGetter<T>::find(uint32_t index) const
    {
      return regions_.recorded(index);
    }

  template <class T>
    inline
    void
    swap(RefGetter<T>& a, RefGetter<T>& b) 
    {
      a.swap(b);
    }
  
  //------------------------------------------------------------
  
  //helper function to make it easier to create a edm::Ref
  
  template<class HandleT>
    typename HandleT::element_type::value_ref
    makeRefToRefGetter(const HandleT& iHandle, typename HandleT::element_type::record_iterator iter) {
    return typename HandleT::element_type::value_ref(iHandle,iter);
  }
  
  //------------------------------------------------------------
  
#if ! GCC_PREREQUISITE(3,4,4)
  // Has swap function
  template <class T>
  struct has_swap<edm::RefGetter<T> > {
    static bool const value = true;
  };

#endif

}
  
#endif

