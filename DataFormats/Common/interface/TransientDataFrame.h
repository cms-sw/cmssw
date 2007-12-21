#ifndef DataFormats_Common_TransientDataFrame_h
#define DataFormats_Common_TransientDataFrame_h

#include "DataFormats/Common/interface/DataFrame.h"
#include <algorithm>

namespace edm {

  /* a fixed size array of 16bit words belonging to
   * a "channel" identified by an 32bit id
   *
   * FIXME interface to be finalized once use-cases fully identified
   * 
   */
  template<unsigned int SIZE>
  class TransientDataFrame {
  public: 
    
    typedef DataFrame::data_type data_type;   
    typedef DataFrame::id_type id_type;   
    
    TransientDataFrame() {}
    TransientDataFrame( id_type i) : m_id(i) {}
    TransientDataFrame(DataFrame const & iframe) : 
      m_id(iframe.id())
    {
      data_type const * p = iframe.begin();
      std::copy(p,p+SIZE,m_data);
    }
    
    int size() const { return SIZE;}
    
    data_type operator[](size_t i) const {
      return m_data[i];
    } 
    
    data_type & operator[](size_t i) {
      return m_data[i];
    } 
    
    id_type id() const { return m_id; }
    
  private:
    id_type m_id;
    data_type m_data[SIZE];
    
  };
  
}

#endif // DataFormats_Common_TransientDataFrame_h
