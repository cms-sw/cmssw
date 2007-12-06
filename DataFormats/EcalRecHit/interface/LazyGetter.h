#ifndef DataFormats_Common_LazyGetter_H
#define DataFormats_Common_LazyGetter_H

#include <algorithm>
#include <vector>
#include <iostream>
#include "boost/concept_check.hpp"
#include "boost/iterator/transform_iterator.hpp"
#include "boost/shared_ptr.hpp"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

  //------------------------------------------------------------

  template <class T> class LazyGetter;
  template <class T> class LazyAdapter;

  //------------------------------------------------------------

  namespace lazydetail {
    inline
      void _throw_range(uint32_t region)
      {
	throw edm::Exception(errors::InvalidReference)
	  << "LazyGetter::"
	  << "find(uint32_t,uint32_t) called with index not in collection;\n"
	  << "index value: " 
	  << region;
      }
  }
  //------------------------------------------------------------

  template<typename T>
    class RegionIndex {
    
    friend class LazyAdapter<T>;

    public:

    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::pair<const_iterator,const_iterator> pair_iterator;

    RegionIndex(uint32_t& region, const_iterator begin, const_iterator end) :
     region_(region), begin_(begin), end_(end), unpacked_(false) {}
    ~RegionIndex() {}
    uint32_t region() const {return region_;}
    const_iterator begin() const {return begin_;}
    const_iterator end() const {return end_;}
    const bool unpacked() const {return unpacked_;}
    pair_iterator find(uint32_t key) const {return std::equal_range(begin_,end_,key);}
   
    private:

    RegionIndex() {}
    void begin(const_iterator newbegin) {begin_ = newbegin;}
    void end(const_iterator newend) {end_ = newend;}
    void unpacked(bool newunpacked) {unpacked_=newunpacked;}
    uint32_t region_;
    const_iterator begin_;
    const_iterator end_;
    bool unpacked_;
  };

  //------------------------------------------------------------

  template<typename T>
    class LazyUnpacker {

    friend class LazyGetter<T>;
    friend class LazyAdapter<T>;

    public: 

    typedef std::vector< RegionIndex<T> > register_type;
    typedef std::vector<T> record_type;

    LazyUnpacker(uint32_t nregions) :
      record_(), register_()
      {
	//At high luminosity:
	//tracker occupancy 1.2%, 3 strip clusters -> ~40,000 clusters.
	//Reserve 100,000 to absorb event-by-event fluctuations.
	record_.reserve(100000); 
	register_.reserve(nregions);
	for (uint32_t iregion=0;iregion<nregions;iregion++) {
	  register_.push_back(RegionIndex<T>(iregion,record().begin(),record().begin()));
	}
      }
    virtual ~LazyUnpacker() {}

    protected:
    virtual void fill(uint32_t&) = 0;
    record_type& record() {return record_;}
    private:
    LazyUnpacker() {}
    record_type record_;
    register_type register_;
  };
  
  //------------------------------------------------------------

  template<typename T>
    struct LazyAdapter : public std::unary_function<const RegionIndex<T>&, const RegionIndex<T>& > {

      /// Constructor with LazyUnpacker
      LazyAdapter(boost::shared_ptr< LazyUnpacker<T> > iGetter) : 
	unpacker_(iGetter) {}

      /// () operator for construction of iterator
      const RegionIndex<T>& operator()(const RegionIndex<T>& region) const {
	if (!region.unpacked()) {
	uint32_t index = region.region();
	unpacker_->register_[index].begin(unpacker_->record().end());
	unpacker_->fill(index); 
	const_cast< RegionIndex<T>& >(region).unpacked(true);
	unpacker_->register_[index].end(unpacker_->record().end());
	}
	return region;
      }

      private:
      boost::shared_ptr<LazyUnpacker<T> > unpacker_;
    };

  //------------------------------------------------------------

  template <class T>
  class LazyGetter 
  {

    BOOST_CLASS_REQUIRE(T, boost, LessThanComparableConcept);

  public:

    typedef std::vector< RegionIndex<T> > register_type;
    typedef std::vector<T> record_type;
    typedef typename register_type::const_iterator register_iterator;
    typedef typename record_type::const_iterator record_iterator;
    typedef boost::transform_iterator< LazyAdapter<T>, typename register_type::const_iterator > const_iterator;
    typedef std::pair<typename record_type::const_iterator, typename record_type::const_iterator> record_pair;
  
    /// Default constructor.
    LazyGetter();
    
    /// Constructor with unpacker.
    LazyGetter(boost::shared_ptr< LazyUnpacker<T> >);

    /// Swap contents of class
    void swap(LazyGetter& other);

    /// Returns true if LazyUnpacker::record_ is empty.
    bool empty() const;

    /// Returns the size of LazyUnpacker::record_.
    uint32_t size() const;

    /// Returns the size of LazyUnpacker::register_.
    uint32_t regions() const;

    /// Returns an iterator to the RegionIndex<T>  for a given index.
    const_iterator find(uint32_t index) const;

    /// Returns a reference to the RegionIndex<T> for a given index.
    const RegionIndex<T>& operator[](uint32_t index) const;

    /// Returns an iterator to the first RegionIndex<T>.
    const_iterator begin() const;

    /// Returns the off-the-end iterator.
    const_iterator end() const;

    /// Returns an iterator to the first RegionIndex<T> without unpacking.
    register_iterator begin_nounpack() const;

    /// Returns the off-the-end iterator.
    register_iterator end_nounpack() const;

    /// Returns boolean describing unpacking status of a given region 
    /// without unpacking
    bool unpacked(uint32_t) const;

  private:

    boost::shared_ptr< LazyUnpacker<T> > unpacker_;
  };

  template <class T>
    inline
    LazyGetter<T>::LazyGetter() : unpacker_()
    {}
  
  template <class T>
    inline
    LazyGetter<T>::LazyGetter(boost::shared_ptr< LazyUnpacker<T> > unpacker) : unpacker_(unpacker) 
    {}

  template <class T>
    inline
    void
    LazyGetter<T>::swap(LazyGetter<T>& other) 
    {
      std::swap(unpacker_,other.unpacker_);
    }
  
  template <class T>
    inline
    bool
    LazyGetter<T>::empty() const 
    {
      return unpacker_->record().empty();
    }
  
  template <class T>
    inline
    uint32_t
    LazyGetter<T>::size() const
    {
      return unpacker_->record().size();
    }
  
  template <class T>
    inline
    uint32_t 
    LazyGetter<T>::regions() const
    {
      return unpacker_->register_.size();
    }

  template <class T>
    inline
    typename LazyGetter<T>::const_iterator
    LazyGetter<T>::find(uint32_t index) const 
    {
      if (index<regions()) return end();
      typename register_type::const_iterator it = unpacker_->register_.begin()+index;
      LazyAdapter<T> adapter(unpacker_);
      return boost::make_transform_iterator(it,adapter);
    }

  template <class T>
    inline
    const RegionIndex<T>&
    LazyGetter<T>::operator[](uint32_t index) const 
    {
      if (index<regions()) edm::lazydetail::_throw_range(index);
      typename register_type::const_iterator it = unpacker_->register_.begin()+index;
      LazyAdapter<T> adapter(unpacker_);
      return *(boost::make_transform_iterator(it,adapter));
    }
  
  template <class T>
    inline
    typename LazyGetter<T>::const_iterator
    LazyGetter<T>::begin() const
    {
      LazyAdapter<T> adapter(unpacker_);
      return boost::make_transform_iterator(unpacker_->register_.begin(),adapter);
    }
  
  template <class T>
    inline
    typename LazyGetter<T>::const_iterator
    LazyGetter<T>::end() const
    {
      LazyAdapter<T> adapter(unpacker_);
      return boost::make_transform_iterator(unpacker_->register_.end(),adapter);
    }
  
  template <class T>
    inline
    typename LazyGetter<T>::register_iterator
    LazyGetter<T>::begin_nounpack() const
    {
      return unpacker_->register_.begin();
    }

  template <class T>
    inline
    typename LazyGetter<T>::register_iterator
    LazyGetter<T>::end_nounpack() const
    {
      return unpacker_->register_.end();
    }

  template <class T>
    inline
    bool 
    LazyGetter<T>::unpacked(uint32_t index) const
    {
      return (index<regions()) ? unpacker_->register_[index].unpacked() : false;
    }

  template <class T>
    inline
    void
    swap(LazyGetter<T>& a, LazyGetter<T>& b) 
    {
      a.swap(b);
    }
  
  //------------------------------------------------------------

  /// Returns RegionIndex for region.
  template<typename T>
    struct FindRegion : public std::binary_function< const LazyGetter<T>&, const uint32_t, const RegionIndex<T>* > {
      typedef FindRegion<T> self;
      typename self::result_type operator()(typename self::first_argument_type iContainer, typename self::second_argument_type iIndex) const {
	return &(*(iContainer.begin()+iIndex));
      }
    };
  
#if ! GCC_PREREQUISITE(3,4,4)
  // has swap function
  template <class T>
    struct has_swap<edm::LazyGetter<T> > {
      static bool const value = true;
    };
#endif
  
}

#endif



