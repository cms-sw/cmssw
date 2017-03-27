#ifndef Cond_BasicPayload_h
#define Cond_BasicPayload_h

#include "CondFormats/Serialization/interface/Serializable.h"

namespace cond {
  
  /** Test class for condition payload
  */
  class BasicPayload {
  public:
    static const size_t vecSize = 100; 
    BasicPayload():m_data0(0.),m_data1(0.),m_vec(vecSize,0){
    }
    BasicPayload( float d0, float d1):m_data0(d0),m_data1(d1),m_vec(vecSize,0){
      for(size_t i=0;i<vecSize;i++) m_vec[i] = d0*i+d1;
    }
    virtual ~BasicPayload(){}

    void print() {
      for ( size_t i=0; i<10; i++ ) {
	for ( size_t j=0;j<10;j++ ){
	  size_t ind = i*10+j;
	  std::cout <<ind<<":"<<m_vec[ind]<<"  ";
	}
	std::cout <<std::endl;
      }
    }
    
  public:
    float m_data0;
    float m_data1;
    std::vector<float> m_vec;
  
  COND_SERIALIZABLE;
};
  
}

#endif
