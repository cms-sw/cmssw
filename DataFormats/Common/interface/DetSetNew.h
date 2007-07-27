#ifndef Common_DetSetNew_H
#define Common_DetSetNew_H

namespace edmNew {

  template<typename T> class DetSetVector;
  
  /* a proxy to a variable size array of T belonging to
   * a "channel" identified by an 32bit id
   *
   * FIXME interface to be finalized once use-cases fully identified
   * 
   */
  template<typename T>
  class DetSet {
  public: 
    typedef DetSetVector<T> Container;
    typedef unsigned int size_type; // for persistency
    typedef unsigned int id_type;
    typedef T data_type;

    typedef data_type * iterator;
    typedef data_type const * const_iterator;

    typedef data_type value_type;
    typedef id_type key_type;
    
    
    inline
    DetSet() : m_id(0), m_data(0), m_size(0){}
    inline
    DetSet(id_type i, data_type const * idata, size_type isize) :
      m_id(i), m_data(idata), m_size(isize) {}
    
    inline
    DetSet(Container const & icont,
	      size_type i);
    inline
    DetSet(Container const & icont,
	   typename Container::Item const & item );

    inline
    void set(Container const & icont,
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
    data_type * data() {
      return const_cast<data_type *>(m_data);
    }
    
    id_type m_id;
    data_type const * m_data;
    size_type m_size;
  };
}

#endif // Common_DetSet_H
