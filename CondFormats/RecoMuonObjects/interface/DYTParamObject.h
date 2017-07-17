#ifndef DytParamObject_h
#define DytParamObject_h

#include<vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/Serialization/interface/Serializable.h"

class DYTParamObject {
 public:
  
  DYTParamObject() { };
  DYTParamObject(uint32_t id, std::vector<double> & params) 
    : m_id(id) , m_params(params) { };
  ~DYTParamObject() { m_params.clear(); };
  
  // Return raw id
  uint32_t id() const {return m_id;};

  // Return param i (i from 0 to size-1)
  double parameter(unsigned int iParam) const;
  
  // Return param vector size (i from 0 to size-1)
  unsigned int paramSize() const { return m_params.size(); };
  
 private:

  uint32_t m_id;
  std::vector<double> m_params;

  COND_SERIALIZABLE;

};

#endif
