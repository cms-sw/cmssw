#ifndef DataFormats_SiStripCommon_SiStripLazyGetter_h
#define DataFormats_SiStripCommon_SiStripLazyGetter_h

#include <algorithm>
#include <vector>
#include <iostream>

#include "boost/concept_check.hpp"
#include "boost/iterator/transform_iterator.hpp"
#include "boost/shared_ptr.hpp"

#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

  //------------------------------------------------------------

  template <class T> class SiStripLazyGetter;
  template <class T> class LazyAdapter;

  //------------------------------------------------------------
  // Helper function, to regularize throwing of exceptions.
  //------------------------------------------------------------

  namespace lgdetail
    {
      // Throw an edm::Exception with an appropriate message
      inline
	void _throw_range(uint32_t i)
	{
	  throw edm::Exception(errors::InvalidReference)
	    << "SiStripLazyGetter:: "
	    << "operator[] called with index not in collection;\n"
	    << "index value: " << i;
	}
    }
  
  template<typename T>
    class RegionIndex {
    
    friend class LazyAdapter<T>;

    public:
    typedef typename std::vector<T>::const_iterator const_iterator;
    RegionIndex(uint32_t& region, const_iterator begin, const_iterator end) :
     region_(region), begin_(begin), end_(end) {}
    ~RegionIndex() {}
    uint32_t region() const {return region_;}
    const_iterator begin() const {return begin_;}
    const_iterator end() const {return end_;}
   
    private:
    RegionIndex() {}
    void begin(const_iterator newbegin) {begin_ = newbegin;}
    void end(const_iterator newend) {end_ = newend;}
    uint32_t region_;
    const_iterator begin_;
    const_iterator end_;
  };

  template<typename T>
    class SiStripLazyUnpacker {

    friend class SiStripLazyGetter<T>;
    friend class LazyAdapter<T>;

    public: 

    typedef RegionIndex<T> register_index;
    typedef std::vector< register_index > register_type;
    typedef std::vector<T> record_type;

    SiStripLazyUnpacker(uint32_t nregions) :
      record_(), register_()
      {
	register_.reserve(nregions);
	//At high luminosity:
	//tracker occupancy 1.2%, 3 strip clusters -> ~40,000 clusters.
	//Reserve 100,000 to absorb event-by-event fluctuations.
	record_.reserve(100000); 
	for (uint32_t iregion=0;iregion<nregions;iregion++) {
	  register_.push_back(RegionIndex<T>(iregion,record().begin(),record().begin()));}
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
  
  template<typename T>
    struct LazyAdapter : public std::unary_function<const RegionIndex<T>&, const RegionIndex<T>& > {

      /// Constructor with SiStripLazyUnpacker
      LazyAdapter(boost::shared_ptr<SiStripLazyUnpacker<T> > iGetter) : 
	unpacker_(iGetter) {}

      /// () operator for construction of iterator
      const RegionIndex<T>& operator()(const RegionIndex<T>& index) const {
	if (index.end() == unpacker_->record().begin()) {
	uint32_t region = index.region();
	unpacker_->register_[region].begin(unpacker_->record().end());
	unpacker_->fill(region);
	unpacker_->register_[region].end(unpacker_->record().end());
	}
	return index;
      }

      private:
    
      boost::shared_ptr<SiStripLazyUnpacker<T> > unpacker_;
    };
  

  template <class T>
  class SiStripLazyGetter 
  {

    BOOST_CLASS_REQUIRE(T, boost, LessThanComparableConcept);

  public:

    typedef RegionIndex<T> register_index;
    typedef std::vector< register_index > register_type;
    typedef std::vector<T> record_type;

    typedef register_index const&  const_reference;
    typedef boost::transform_iterator< LazyAdapter<T>, typename register_type::const_iterator > const_iterator;
    typedef typename register_type::size_type size_type;

    SiStripLazyGetter() {}
    
    SiStripLazyGetter(boost::shared_ptr< SiStripLazyUnpacker<T> > iGetter) :
      unpacker_(iGetter) {}

    void swap(SiStripLazyGetter& other);

    /// Return true if SiStripLazyUnpacker::record_ is empty.
    bool empty() const;

    /// Return the size of SiStripLazyUnpacker::record_.
    size_type size() const;

    /// Return an iterator to the SiStripLazyUnpacker<T>::register_ for a 
    /// given Region id, or end() if there is no such Region.
    const_iterator find(uint32_t region) const;

    /// Return a reference to the SiStripLazyUnpacker<T>::register_ for a 
    /// given Region id, or throw an edm::Exception if there is no such 
    /// Region.
    const_reference operator[](uint32_t region) const;

    /// Return an iterator to the first element of 
    /// SiStripLazyUnpacker<T>::register_.
    const_iterator begin() const;

    /// Return the off-the-end iterator of 
    /// SiStripLazyUnpacker<T>::register_ .
    const_iterator end() const;

  private:
    boost::shared_ptr< SiStripLazyUnpacker<T> > unpacker_;
  };

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
  typename SiStripLazyGetter<T>::size_type
  SiStripLazyGetter<T>::size() const
  {
    return unpacker_->record().size();
  }

  template <class T>
  inline
  typename SiStripLazyGetter<T>::const_iterator
  SiStripLazyGetter<T>::find(uint32_t region) const
  {
    typename register_type::const_iterator it;
    if (unpacker_->register_.size() < region+1) it = unpacker_->register_.end();
    else it = unpacker_->register_.begin()+region;
    LazyAdapter<T> adapter(unpacker_);
    return boost::make_transform_iterator(it,adapter);
  }

  template <class T>
  inline
  typename SiStripLazyGetter<T>::const_reference
  SiStripLazyGetter<T>::operator[](uint32_t region) const 
  {
    const_iterator it = this->find(region);
    if (it == this->end()) lgdetail::_throw_range(region);
    return *it;
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
  void
  swap(SiStripLazyGetter<T>& a, SiStripLazyGetter<T>& b) 
  {
    a.swap(b);
  }

#if ! GCC_PREREQUISITE(3,4,4)
  // has swap function
  template <class T>
  struct has_swap<edm::SiStripLazyGetter<T> > {
    static bool const value = true;
  };
#endif

}

#endif



