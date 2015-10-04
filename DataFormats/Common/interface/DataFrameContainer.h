#ifndef DataFormats_Common_DataFrameContainer_h
#define DataFormats_Common_DataFrameContainer_h

#include "DataFormats/Common/interface/DataFrame.h"

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include<vector>
#include<algorithm>

class TestDataFrame;

namespace edm {

  /** an optitimized container that linearized a "vector of vector".
   *  It corresponds to a set of fixed size array of 16bit words each belonging
   *  to a "channel" identified by an 32bit id
   *
   * FIXME interface to be finalized once use-cases fully identified
   *
   * although it can be sorted internally it is strongly adviced to
   * fill it already sorted....
   *
   */
  class DataFrameContainer {
  public:
    typedef unsigned int size_type; // for persistency
    typedef unsigned int id_type;
    typedef unsigned short data_type;
    typedef std::vector<id_type> IdContainer;
    typedef std::vector<data_type> DataContainer;
    typedef std::vector<id_type>::iterator IdIter;
    typedef std::vector<data_type>::iterator DataIter;
    typedef std::pair<IdIter,DataIter> IterPair;
    typedef std::vector<id_type>::const_iterator const_IdIter;
    typedef std::vector<data_type>::const_iterator const_DataIter;
    typedef std::pair<const_IdIter,const_DataIter> const_IterPair;
    
    struct IterHelp {
      typedef DataFrame result_type;
      IterHelp(DataFrameContainer const & iv) : v(iv){}
      
      DataFrame const & operator()(int i) const {
	frame.set(v,i);
	return frame;
      } 
    private:
      DataFrameContainer const & v;
      mutable DataFrame frame;
    };
    
    typedef boost::transform_iterator<IterHelp,boost::counting_iterator<int> > const_iterator;
    
    DataFrameContainer() :
      m_subdetId(0), m_stride(0),
      m_ids(), m_data(){}
    
    explicit DataFrameContainer(size_t istride, int isubdet=0, size_t isize=0) :
      m_subdetId(isubdet), m_stride(istride),
      m_ids(isize), m_data(isize*m_stride){}
    
    void swap(DataFrameContainer & rh) {
      std::swap(m_subdetId,rh.m_subdetId);
      std::swap(m_stride,rh.m_stride);
      m_ids.swap(rh.m_ids);
      m_data.swap(rh.m_data);
    }
    
    DataFrameContainer& operator=(DataFrameContainer const& rhs) {
      DataFrameContainer temp(rhs);
      this->swap(temp);
      return *this;
    }

    void swap(IdContainer & iic, DataContainer & idc) {
      m_ids.swap(iic);
      m_data.swap(idc);
    }
    
    void reserve(size_t isize) {
      m_ids.reserve(isize);
      m_data.reserve(isize*m_stride);
    }
    
    void resize(size_t isize) {
      m_ids.resize(isize);
      m_data.resize(isize*m_stride);
    }

    void sort();
    
    // FIXME not sure what the best way to add one cell to cont
    void push_back(id_type iid, data_type const * idata) {
      m_ids.push_back(iid);
      size_t cs = m_data.size();
      m_data.resize(m_data.size()+m_stride);
      std::copy(idata,idata+m_stride,m_data.begin()+cs);
    }
    //make space for it
    void push_back(id_type iid) {
      m_ids.push_back(iid);
      m_data.resize(m_data.size()+m_stride);
    }
    // overwrite back (very ad hoc interface...)
    void set_back(id_type iid, data_type const * idata) {
      m_ids.back() = iid;
      size_t cs = m_data.size()-m_stride;
      std::copy(idata,idata+m_stride,m_data.begin()+cs);
    }
    void set_back(id_type iid) {
      m_ids.back() = iid;
    }
    void set_back(data_type const * idata) {
      size_t cs = m_data.size()-m_stride;
      std::copy(idata,idata+m_stride,m_data.begin()+cs);
    }

    DataFrame back() {
      return DataFrame(*this,size()-1);
    }

    void pop_back() {
      m_ids.resize(m_ids.size()-1);
      m_data.resize(m_data.size()-m_stride);
    }

    //---------------------------------------------------------
    
    IterPair pair(size_t i) {
      return IterPair(m_ids.begin()+i,m_data.begin()+i*m_stride);
    }
    
    const_IterPair pair(size_t i) const {
      return const_IterPair(m_ids.begin()+i,m_data.begin()+i*m_stride);
    }
    
    DataFrame operator[](size_t i) {
      return DataFrame(*this,i);
    }
    
    DataFrame operator[](size_t i) const {
      return DataFrame(*this,i);
    }
    
    // slow interface
    const_iterator find(id_type i) const {
      const_IdIter p = std::lower_bound(m_ids.begin(),m_ids.end(),i);
      return (p==m_ids.end() || (*p)!=i) ? end() :
	boost::make_transform_iterator(boost::counting_iterator<int>(p-m_ids.begin()),
				       IterHelp(*this));
    }
    
    const_iterator begin() const {
      return  boost::make_transform_iterator(boost::counting_iterator<int>(0),
					     IterHelp(*this));
    }
    const_iterator end() const {
      return  boost::make_transform_iterator(boost::counting_iterator<int>(size()),
					     IterHelp(*this));
    }
    
    
    int subdetId() const { return m_subdetId; }

    size_type stride() const { return m_stride; }
    
    bool empty() const { return m_ids.empty();}

    size_type size() const { return m_ids.size();}
    
    data_type operator()(size_t cell, size_t frame) const {
      return m_data[cell*m_stride+frame];
    }
    
    data_type const * frame(size_t cell) const {
      return &m_data[cell*m_stride];
    }
    
    id_type id(size_t cell) const {
      return m_ids[cell];
    }
    
    // IdContainer const & ids() const { return m_ids;}
    // DataContainer const & data() const { return  m_data;}
    
  private:
    //for testing
    friend class ::TestDataFrame;
    
    // subdetector id (as returned by  DetId::subdetId())
    int m_subdetId;

    // can be a enumerator, or a template argument
    size_type m_stride;
    
    IdContainer m_ids;
    DataContainer m_data;
    
  };
  
  inline
  DataFrame::DataFrame(DataFrameContainer const & icont,
		       size_type i) :
    m_id(icont.id(i)), m_data(icont.frame(i)), m_size(icont.stride()){}

  inline
  void DataFrame::set(DataFrameContainer const & icont,
		      size_type i) {
    m_id=icont.id(i); 
    m_data=icont.frame(i);
    m_size=icont.stride();
  }
  
  // Free swap function
  inline
  void
  swap(DataFrameContainer& lhs, DataFrameContainer& rhs) {
    lhs.swap(rhs);
  }

}

// The standard allows us to specialize std::swap for non-templates.
// This ensures that DataFrameContainer::swap() will be used in algorithms.

namespace std {
  template <> inline void swap(edm::DataFrameContainer& lhs, edm::DataFrameContainer& rhs) {  
    lhs.swap(rhs);
  }
}

#endif // DataFormats_Common_DataFrameContainer_h
