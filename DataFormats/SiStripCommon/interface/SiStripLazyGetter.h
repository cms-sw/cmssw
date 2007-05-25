#ifndef DataFormats_SiStripCommon_SiStripLazyGetter_h
#define DataFormats_SiStripCommon_SiStripLazyGetter_h

#include <algorithm>
#include <vector>

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
    class SiStripLazyUnpacker {

    friend class SiStripLazyGetter<T>;
    friend class LazyAdapter<T>;

    public: 
    typedef std::vector<T> Record;
    typedef typename Record::const_iterator const_iterator;
    typedef std::pair<const_iterator,const_iterator> IndexPair;
    typedef std::vector< IndexPair > Index;

    SiStripLazyUnpacker(uint32_t nregions) :
      record_(), index_()
      {
	index_.reserve(nregions);
	//At high luminosity:
	//tracker occupancy 1.2%, 3 strip clusters -> ~40,000 clusters.
	//Reserve 100,000 to absorb event-by-event fluctuations.
	record_.reserve(100000); 
	for (uint32_t iregion=0;iregion<nregions;iregion++) {
	  index_.push_back(IndexPair(record_.begin()-iregion-1,
				     record_.begin()-iregion-1));
	}
      }
    virtual ~SiStripLazyUnpacker() {}

    protected:
    virtual void fill(uint32_t&) = 0;
    Record& record() {return record_;}
    private:
    SiStripLazyUnpacker() {}
    Record record_;
    Index index_;
  };
  
  template<typename T>
    struct LazyAdapter : public std::unary_function<const typename SiStripLazyUnpacker<T>::IndexPair&, const typename SiStripLazyUnpacker<T>::IndexPair& > {

      /// Constructor with SiStripLazyUnpacker
      LazyAdapter(boost::shared_ptr<SiStripLazyUnpacker<T> > iGetter) : 
	getter_(iGetter) {}

      /// () operator for construction of iterator
      const typename SiStripLazyUnpacker<T>::IndexPair& operator()(const typename SiStripLazyUnpacker<T>::IndexPair& ipair) const {
	int diff = getter_->record().begin() - ipair.first;
	if (diff>0) {
	  uint32_t region = (uint32_t)(diff-1);
	  getter_->index_[region].first = getter_->record_.end();
	  getter_->fill(region);
   	  getter_->index_[region].second = getter_->record_.end();
	}
	return ipair;
      }

      private:
      /// Data members
      boost::shared_ptr<SiStripLazyUnpacker<T> > getter_;
    };
  

  template <class T>
  class SiStripLazyGetter 
  {

    BOOST_CLASS_REQUIRE(T, boost, LessThanComparableConcept);

  public:

    typedef std::vector< typename SiStripLazyUnpacker<T>::IndexPair > collection_type;
    typedef typename SiStripLazyUnpacker<T>::IndexPair const&  const_reference;
    typedef boost::transform_iterator< LazyAdapter<T>, typename collection_type::const_iterator > const_iterator;
    typedef typename collection_type::size_type size_type;

    SiStripLazyGetter() {}
    
    SiStripLazyGetter(boost::shared_ptr< SiStripLazyUnpacker<T> > iGetter) :
      getter_(iGetter) {}

    void swap(SiStripLazyGetter& other);

    /// Return true if SiStripLazyUnpacker::record_ is empty.
    bool empty() const;

    /// Return the size of SiStripLazyUnpacker::record_.
    size_type size() const;

    /// Return an iterator to the SiStripLazyUnpacker<T>::index_ for a 
    /// given Region id, or end() if there is no such Region.
    const_iterator find(uint32_t region) const;

    /// Return a reference to the SiStripLazyUnpacker<T>::index_ for a 
    /// given Region id, or throw an edm::Exception if there is no such 
    /// Region.
    const_reference operator[](uint32_t region) const;

    /// Return an iterator to the first element of 
    /// SiStripLazyUnpacker<T>::index_.
    const_iterator begin() const;

    /// Return the off-the-end iterator of 
    /// SiStripLazyUnpacker<T>::index_ .
    const_iterator end() const;

  private:
    boost::shared_ptr< SiStripLazyUnpacker<T> > getter_;
  };

  template <class T>
  inline
  void
  SiStripLazyGetter<T>::swap(SiStripLazyGetter<T>& other) 
  {
    std::swap(getter_,other.getter_);
  }

  template <class T>
  inline
  bool
  SiStripLazyGetter<T>::empty() const 
  {
    return getter_->record().empty();
  }

  template <class T>
  inline
  typename SiStripLazyGetter<T>::size_type
  SiStripLazyGetter<T>::size() const
  {
    return getter_->record().size();
  }

  template <class T>
  inline
  typename SiStripLazyGetter<T>::const_iterator
  SiStripLazyGetter<T>::find(uint32_t region) const
  {
    typename collection_type::const_iterator it;
    if (getter_->index_.size() < region+1) it = getter_->index_.end();
    else it = getter_->index_.begin()+region;
    LazyAdapter<T> adapter(getter_);
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
    LazyAdapter<T> adapter(getter_);
    return boost::make_transform_iterator(getter_->index_.begin(),adapter);
  }

  template <class T>
  inline
  typename SiStripLazyGetter<T>::const_iterator
  SiStripLazyGetter<T>::end() const
  {
    LazyAdapter<T> adapter(getter_);
    return boost::make_transform_iterator(getter_->index_.end(),adapter);
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

