#ifndef DataFormats_Common_DataFrame_h
#define DataFormats_Common_DataFrame_h

class TestDataFrame;
namespace edm {

  class DataFrameContainer;
  
  /* a proxy to a fixed size array of 16bit words belonging to
   * a "channel" identified by an 32bit id
   *
   * FIXME interface to be finalized once use-cases fully identified
   * 
   */
  class DataFrame {
  public: 
    
    typedef unsigned int size_type; // for persistency
    typedef unsigned int id_type;
    typedef unsigned short data_type;
    typedef data_type * iterator;
    typedef data_type const * const_iterator;
    
    
    inline
    DataFrame() : m_id(0), m_data(0), m_size(0){}
    inline
    DataFrame(id_type i, data_type const * idata, size_type isize) :
      m_id(i), m_data(idata), m_size(isize) {}
    
    inline
    DataFrame(DataFrameContainer const & icont,
	      size_type i);
    inline
    void set(DataFrameContainer const & icont,
	     size_type i);
    inline
    data_type & operator[](size_type i) {
      return data()[i];
    }
    
    inline
    data_type operator[](size_type i) const {
      return m_data[i];
    }
    
    inline
    iterator begin() { return data();}

    inline
    iterator end() { return data()+m_size;}

    inline
    const_iterator begin() const { return m_data;}

    inline
    const_iterator end() const { return m_data+m_size;}


    inline
    id_type id() const { return m_id;}
    
    inline
    size_type size() const { return m_size; }
    
  private:
    //for testing
    friend class ::TestDataFrame;

    data_type * data() {
      return const_cast<data_type *>(m_data);
    }
    
    id_type m_id;
    data_type const * m_data;
    size_type m_size;
  };
}

#endif // DataFormats_Common_DataFrame_h
