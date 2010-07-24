#ifndef DataFormats_Common_LazyGetter_h
#define DataFormats_Common_LazyGetter_h

#include <algorithm>
#include <vector>
#include "boost/concept_check.hpp"
#include "boost/iterator/transform_iterator.hpp"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  //------------------------------------------------------------

  template <class T> class LazyAdapter;
  template <class T> class LazyGetter;
  template <class T> class FindValue;

  //------------------------------------------------------------

  namespace lazydetail {
    inline
      void _throw_range(uint32_t region)
      {
	Exception::throwThis(errors::InvalidReference,
	  "LazyGetter::"
	  "find(uint32_t,uint32_t) called with index not in collection;\n"
	  "index value: ",
	  region);
      }
  }
  //------------------------------------------------------------

  template<class T>
    class RegionIndex {

    friend class LazyAdapter<T>;

    public:

    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef std::pair<const_iterator,const_iterator> pair_iterator;

    /// Default constructor
    RegionIndex();

    /// Constructor
    RegionIndex(uint32_t region, uint32_t start, uint32_t finish, const LazyGetter<T>* theLazyGetter);

    /// Get region number
    uint32_t region() const;

    /// Get start index
    uint32_t start() const;

    /// Get off-the-end finish index
    uint32_t finish() const;

    /// Get unpacking status
    bool unpacked() const;

    /// Get begin iterator
    const_iterator begin() const;

    /// Get off the end iterator
    const_iterator end() const;

    /// Update the pointer to the lazyGetter
    RegionIndex<T>& updateLazyGetter(const LazyGetter<T>* newLazyGetter);

    /// Get range of T on on det
    pair_iterator find(uint32_t id) const;

    private:

    /// Set start index
    void start(uint32_t);

    /// Set of-the-end finish index
    void finish(uint32_t);

    /// Set unpacking status
    void unpacked(bool);

    uint32_t region_;
    uint32_t start_;
    uint32_t finish_;
    bool unpacked_;
    const LazyGetter<T> * getter_;
  };

  //------------------------------------------------------------

  template <class T>
    inline
    RegionIndex<T>::RegionIndex() :
    region_(0),
    start_(0),
    finish_(0),
    unpacked_(false),
    getter_(NULL)
      {}

  template <class T>
    inline
    RegionIndex<T>::RegionIndex(uint32_t region, uint32_t start, uint32_t finish, const LazyGetter<T>* theLazyGetter) :
    region_(region),
    start_(start),
    finish_(finish),
    unpacked_(false),
    getter_(theLazyGetter)
      {}

  template <class T>
    inline
    uint32_t
    RegionIndex<T>::region() const
    {
      return region_;
    }

  template <class T>
    inline
    uint32_t
    RegionIndex<T>::start() const
    {
      return start_;
    }

  template <class T>
    inline
    uint32_t RegionIndex<T>::finish() const
    {
      return finish_;
    }

  template <class T>
    inline
    bool
    RegionIndex<T>::unpacked() const
    {
      return unpacked_;
    }

  template <class T>
    inline
    void
    RegionIndex<T>::start(uint32_t newstart)
    {
      start_=newstart;
    }

  template <class T>
    inline
    void
    RegionIndex<T>::finish(uint32_t newfinish)
    {
      finish_=newfinish;
    }

  template <class T>
    inline
    void
    RegionIndex<T>::unpacked(bool newunpacked)
    {
      unpacked_=newunpacked;
    }

 template <class T>
    inline
   typename RegionIndex<T>::const_iterator
   RegionIndex<T>::begin() const
   {
     //check pointer here and throw if null
     return getter_->begin_record()+start_;
   }

  template <class T>
    inline
    typename RegionIndex<T>::const_iterator
    RegionIndex<T>::end() const
    {
      //check pointer here and throw if null
      return getter_->begin_record()+finish_;
    }

  template <class T>
    inline
    RegionIndex<T>&
    RegionIndex<T>::updateLazyGetter(const LazyGetter<T>* newGetter)
    {
      getter_ = newGetter;
      return *this;
    }

  template <class T>
    inline
    typename RegionIndex<T>::pair_iterator
    RegionIndex<T>::find(uint32_t id) const
    {
      return std::equal_range(begin(),end(),id);
    }

  //------------------------------------------------------------

  template<typename T>
    class LazyUnpacker {
    public:
    typedef std::vector<T> record_type;
    virtual void fill(const uint32_t&, record_type&)=0;
    virtual ~LazyUnpacker() {}
  };

  //------------------------------------------------------------

  template<typename T>
    class LazyAdapter : public std::unary_function<const RegionIndex<T>&, const RegionIndex<T>& > {
      public:

      typedef std::vector<T> record_type;

      /// Constructor
      LazyAdapter(const LazyUnpacker<T>*,const record_type*, const LazyGetter<T>*);

      /// () operator for construction of iterator
      const RegionIndex<T>& operator()(const RegionIndex<T>& region) const;

      private:
      LazyUnpacker<T>* unpacker_;
      record_type * record_;
      const LazyGetter<T>* getter_;
    };

    template <class T>
    inline
      LazyAdapter<T>::LazyAdapter(const LazyUnpacker<T>* iunpacker, const record_type* irecord, const LazyGetter<T>* getter) :
      unpacker_(const_cast< LazyUnpacker<T>* >(iunpacker)),
      record_(const_cast<record_type*>(irecord)),
      getter_(getter) {}

  template <class T>
    inline
    const RegionIndex<T>&
    LazyAdapter<T>::operator()(const RegionIndex<T>& index) const
    {
      RegionIndex<T>& indexref = const_cast< RegionIndex<T>& >(index);
      if (!index.unpacked()) {
	indexref.start(record_->size());
	unpacker_->fill(index.region(),*record_);
	indexref.unpacked(true);
	indexref.finish(record_->size());
        indexref.updateLazyGetter(getter_);
      }
      return index;
    }

  //------------------------------------------------------------

  template<typename T> class UpdateGetterAdapter : public std::unary_function<const RegionIndex<T>&, const RegionIndex<T>& > {

  public:

    /// Constructor
    UpdateGetterAdapter(const LazyGetter<T>*);

    /// () operator for construction of iterator
    const RegionIndex<T>& operator()(const RegionIndex<T>&) const;

  private:

    const LazyGetter<T>* getter_;
  };

  template <class T>
    inline
    UpdateGetterAdapter<T>::UpdateGetterAdapter(const LazyGetter<T>* getter)
    : getter_(getter) {}

  template <class T>
    inline
    const RegionIndex<T>&
    UpdateGetterAdapter<T>::operator()(const RegionIndex<T>& index) const
    {
      RegionIndex<T>& indexref = const_cast< RegionIndex<T>& >(index);
      indexref.updateLazyGetter(getter_);
      return index;
    }

  //------------------------------------------------------------

  template <class T>
  class LazyGetter
  {

    BOOST_CLASS_REQUIRE(T, boost, LessThanComparableConcept);

  public:

    typedef std::vector< RegionIndex<T> > register_type;
    typedef std::vector<T> record_type;
    typedef boost::transform_iterator< UpdateGetterAdapter<T>, typename register_type::const_iterator > register_iterator;
    typedef typename record_type::const_iterator record_iterator;
    typedef boost::transform_iterator< LazyAdapter<T>, typename register_type::const_iterator > const_iterator;
    typedef Ref< LazyGetter<T>, T, FindValue<T> > value_ref;

    /// Default constructor.
    LazyGetter();

    /// Constructor with unpacker.
    LazyGetter(uint32_t, const boost::shared_ptr< LazyUnpacker<T> >&);

    /// Returns the size of LazyUnpacker::register_.
    uint32_t regions() const;

    /// Returns an iterator to the register_ for a given index.
    const_iterator find(uint32_t index) const;

    /// Returns a reference to the register_ for a given index.
    const RegionIndex<T>& operator[](uint32_t index) const;

    /// Returns an iterator to the first register_
    const_iterator begin() const;

    /// Returns the off-the-end iterator.
    const_iterator end() const;

    /// Returns an iterator to the start of the register_ without unpacking.
    register_iterator begin_nounpack() const;

    /// Returns the off-the-end iter.
    register_iterator end_nounpack() const;

    /// Returns boolean describing unpacking status of a given region
    /// without unpacking
    bool unpacked(uint32_t) const;

    /// Returns an iterator to the start of record_.
    record_iterator begin_record() const;

    /// Returns an off-the-end iterator.
    record_iterator end_record() const;

    /// Returns the size of the record_.
    uint32_t size() const;

    /// Returns true if record_ is empty.
    bool empty() const;

    /// Swap contents of class
    void swap(LazyGetter& other);

  private:

    boost::shared_ptr< LazyUnpacker<T> > unpacker_;
    std::vector<T> record_;
    std::vector< RegionIndex<T> > register_;
  };

  template <class T>
    inline
    LazyGetter<T>::LazyGetter() : unpacker_(), record_(), register_()
    {}

  template <class T>
    inline
    LazyGetter<T>::LazyGetter(uint32_t nregions,const boost::shared_ptr< LazyUnpacker<T> > & unpacker) : unpacker_(unpacker), record_(), register_()
    {
      //Reserve 100,000 to absorb event-by-event fluctuations.
      record_.reserve(100000);
      register_.reserve(nregions);
      for (uint32_t iregion=0;iregion<nregions;iregion++) {
	register_.push_back(RegionIndex<T>(iregion,0,0,this));
      }
    }

  template <class T>
    inline
    void
    LazyGetter<T>::swap(LazyGetter<T>& other)
    {
      std::swap(unpacker_,other.unpacker_);
      std::swap(record_,other.record_);
      std::swap(register_,other.register_);
    }

  template <class T>
    inline
    uint32_t
    LazyGetter<T>::regions() const
    {
      return register_.size();
    }

  template <class T>
    inline
    typename LazyGetter<T>::const_iterator
    LazyGetter<T>::find(uint32_t index) const
    {
      if (index>=regions()) return end();
      typename register_type::const_iterator it = register_.begin()+index;
      const LazyAdapter<T> adapter(unpacker_.get(),&record_,this);
      return boost::make_transform_iterator(it,adapter);
    }

  template <class T>
    inline
    const RegionIndex<T>&
    LazyGetter<T>::operator[](uint32_t index) const
    {
      if (index>=regions()) edm::lazydetail::_throw_range(index);
      typename register_type::const_iterator it = register_.begin()+index;
      const LazyAdapter<T> adapter(unpacker_.get(),&record_,this);
      return *(boost::make_transform_iterator(it,adapter));
    }

  template <class T>
    inline
    typename LazyGetter<T>::const_iterator
    LazyGetter<T>::begin() const
    {
      const LazyAdapter<T> adapter(unpacker_.get(),&record_,this);
      return boost::make_transform_iterator(register_.begin(),adapter);
    }

  template <class T>
    inline
    typename LazyGetter<T>::const_iterator
    LazyGetter<T>::end() const
    {
      const LazyAdapter<T> adapter(unpacker_.get(),&record_,this);
      return boost::make_transform_iterator(register_.end(),adapter);
    }

  template <class T>
    inline
    typename LazyGetter<T>::register_iterator
    LazyGetter<T>::begin_nounpack() const
    {
      const UpdateGetterAdapter<T> adapter(this);
      return boost::make_transform_iterator(register_.begin(),adapter);
    }

  template <class T>
    inline
    typename LazyGetter<T>::register_iterator
    LazyGetter<T>::end_nounpack() const
    {
      const UpdateGetterAdapter<T> adapter(this);
      return boost::make_transform_iterator(register_.end(),adapter);
    }

  template <class T>
    inline
    bool
    LazyGetter<T>::unpacked(uint32_t index) const
    {
      return (index<regions()) ? register_[index].unpacked() : false;
    }

  template <class T>
    inline
    typename LazyGetter<T>::record_iterator
    LazyGetter<T>::begin_record() const
    {
      return record_.begin();
    }

  template <class T>
    inline
    typename LazyGetter<T>::record_iterator
    LazyGetter<T>::end_record() const
    {
      return record_.end();
    }

  template <class T>
    inline
    uint32_t
    LazyGetter<T>::size() const
    {
      return record_.size();
    }

  template <class T>
    inline
    bool
    LazyGetter<T>::empty() const
    {
      return record_.empty();
    }

  template <class T>
    inline
    void
    swap(LazyGetter<T>& a, LazyGetter<T>& b)
    {
      a.swap(b);
    }

  //------------------------------------------------------------


  //------------------------------------------------------------

  template<typename T> struct FindRegion : public std::binary_function< const LazyGetter<T>&, const uint32_t, const RegionIndex<T>* > {
    typename FindRegion<T>::result_type operator()(typename FindRegion<T>::first_argument_type iContainer, typename FindRegion<T>::second_argument_type iIndex) {
      //return &(const_cast< RegionIndex<T>& >(*(const_cast< LazyGetter<T>& >(iContainer).begin()+iIndex)).updateLazyGetter(&iContainer));
      return &(const_cast< RegionIndex<T>& >(*(const_cast< LazyGetter<T>& >(iContainer).begin()+iIndex)));
    }
  };

  //------------------------------------------------------------

  template<typename T> struct FindValue : public std::binary_function< const LazyGetter<T>&, const uint32_t, const T* > {
    typename FindValue<T>::result_type operator()(typename FindValue<T>::first_argument_type container, typename FindValue<T>::second_argument_type index) const {return &*(container.begin_record()+index);}
  };

  //------------------------------------------------------------

  template<typename T> Ref< LazyGetter<T>, T, FindValue<T> >
    makeRefToLazyGetter(const Handle< LazyGetter<T> >& handle, const uint32_t index) {return Ref< LazyGetter<T>, T, FindValue<T> >(handle,index);}

  //------------------------------------------------------------

}

#endif



