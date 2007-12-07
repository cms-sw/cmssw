#ifndef DataFormats_SiStripCommon_SiStripLazyGetter_H
#define DataFormats_SiStripCommon_SiStripLazyGetter_H

#include <algorithm>
#include <vector>
#include <iostream>
#include "boost/concept_check.hpp"
#include "boost/iterator/transform_iterator.hpp"
#include "boost/shared_ptr.hpp"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

  //------------------------------------------------------------

  template <class T> class SiStripLazyGetter;
  template <class T> class LazyAdapter;

  //------------------------------------------------------------

  namespace sistripdetail
    {
      inline
	void _throw_range(uint32_t region)
	{
	  throw edm::Exception(errors::InvalidReference)
	    << "SiStripLazyGetter::"
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
    class SiStripLazyUnpacker {

    friend class SiStripLazyGetter<T>;
    friend class LazyAdapter<T>;

    public: 

    typedef std::vector< RegionIndex<T> > register_type;
    typedef std::vector<T> record_type;

    SiStripLazyUnpacker(uint32_t nregions) :
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
    virtual ~SiStripLazyUnpacker() {}

    protected:
    virtual void fill(uint32_t&) = 0;
    record_type& record() {return record_;}
    private:
    SiStripLazyUnpacker() {}
    record_type record_;
    register_type register_;
  };
  
  //------------------------------------------------------------

  template<typename T>
    struct LazyAdapter : public std::unary_function<const RegionIndex<T>&, const RegionIndex<T>& > {

      /// Constructor with SiStripLazyUnpacker
      LazyAdapter(boost::shared_ptr< SiStripLazyUnpacker<T> > iGetter) : 
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
      boost::shared_ptr<SiStripLazyUnpacker<T> > unpacker_;
    };

  //------------------------------------------------------------

  template <class T>
  class SiStripLazyGetter 
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
    SiStripLazyGetter();
    
    /// Constructor with unpacker.
    SiStripLazyGetter(boost::shared_ptr< SiStripLazyUnpacker<T> >);

    /// Swap contents of class
    void swap(SiStripLazyGetter& other);

    /// Returns true if SiStripLazyUnpacker::record_ is empty.
    bool empty() const;

    /// Returns the size of SiStripLazyUnpacker::record_.
    uint32_t size() const;

    /// Returns the size of SiStripLazyUnpacker::register_.
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

    boost::shared_ptr< SiStripLazyUnpacker<T> > unpacker_;
  };

  template <class T>
    inline
    SiStripLazyGetter<T>::SiStripLazyGetter() : unpacker_()
    {}
  
  template <class T>
    inline
    SiStripLazyGetter<T>::SiStripLazyGetter(boost::shared_ptr< SiStripLazyUnpacker<T> > unpacker) : unpacker_(unpacker) 
    {}

  template <class T>
    inline
    void
    SiStripLazyGetter<T>::swap(SiStripLazyGetter<T>& other) 
    {
      std::swap(unpacker_,other.unpacker_);
    }
  
  template <class T>
    inline
    bool
    SiStripLazyGetter<T>::empty() const 
    {
      return unpacker_->record().empty();
    }
  
  template <class T>
    inline
    uint32_t
    SiStripLazyGetter<T>::size() const
    {
      return unpacker_->record().size();
    }
  
  template <class T>
    inline
    uint32_t 
    SiStripLazyGetter<T>::regions() const
    {
      return unpacker_->register_.size();
    }

  template <class T>
    inline
    typename SiStripLazyGetter<T>::const_iterator
    SiStripLazyGetter<T>::find(uint32_t index) const 
    {
      if (index<regions()) return end();
      typename register_type::const_iterator it = unpacker_->register_.begin()+index;
      LazyAdapter<T> adapter(unpacker_);
      return boost::make_transform_iterator(it,adapter);
    }

  template <class T>
    inline
    const RegionIndex<T>&
    SiStripLazyGetter<T>::operator[](uint32_t index) const 
    {
      if (index<regions()) sistripdetail::_throw_range(index);
      typename register_type::const_iterator it = unpacker_->register_.begin()+index;
      LazyAdapter<T> adapter(unpacker_);
      return *(boost::make_transform_iterator(it,adapter));
    }
  
  template <class T>
    inline
    typename SiStripLazyGetter<T>::const_iterator
    SiStripLazyGetter<T>::begin() const
    {
      LazyAdapter<T> adapter(unpacker_);
      return boost::make_transform_iterator(unpacker_->register_.begin(),adapter);
    }
  
  template <class T>
    inline
    typename SiStripLazyGetter<T>::const_iterator
    SiStripLazyGetter<T>::end() const
    {
      LazyAdapter<T> adapter(unpacker_);
      return boost::make_transform_iterator(unpacker_->register_.end(),adapter);
    }
  
  template <class T>
    inline
    typename SiStripLazyGetter<T>::register_iterator
    SiStripLazyGetter<T>::begin_nounpack() const
    {
      return unpacker_->register_.begin();
    }

  template <class T>
    inline
    typename SiStripLazyGetter<T>::register_iterator
    SiStripLazyGetter<T>::end_nounpack() const
    {
      return unpacker_->register_.end();
    }

  template <class T>
    inline
    bool 
    SiStripLazyGetter<T>::unpacked(uint32_t index) const
    {
      return (index<regions()) ? unpacker_->register_[index].unpacked() : false;
    }

  template <class T>
    inline
    void
    swap(SiStripLazyGetter<T>& a, SiStripLazyGetter<T>& b) 
    {
      a.swap(b);
    }
  
  //------------------------------------------------------------

  /// Returns RegionIndex for region.
  template<typename T>
    struct FindRegion : public std::binary_function< const SiStripLazyGetter<T>&, const uint32_t, const RegionIndex<T>* > {
      typedef FindRegion<T> self;
      typename self::result_type operator()(typename self::first_argument_type iContainer, typename self::second_argument_type iIndex) const {
	return &(*(iContainer.begin()+iIndex));
      }
    };
  
#if ! GCC_PREREQUISITE(3,4,4)
  // has swap function
  template <class T>
    struct has_swap<edm::SiStripLazyGetter<T> > {
      static bool const value = true;
    };
#endif
  
}

#endif



