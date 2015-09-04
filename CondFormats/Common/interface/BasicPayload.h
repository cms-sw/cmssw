#ifndef Cond_BasicPayload_h
#define Cond_BasicPayload_h

#include "CondFormats/Serialization/interface/Serializable.h"

namespace cond {
  
  /** Test class for condition payload
  */
  class BasicPayload {
  public:
    
    BasicPayload():m_data0(0.),m_data1(0.){}
    BasicPayload( float d0, float d1):m_data0(d0),m_data1(d1){}
    virtual ~BasicPayload(){}
    
  public:
    float m_data0;
    float m_data1;
  
  COND_SERIALIZABLE;
};
  
}

#endif
