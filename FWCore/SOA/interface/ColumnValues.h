#ifndef FWCore_SOA_ColumnValues_h
#define FWCore_SOA_ColumnValues_h
// -*- C++ -*-
//
// Package:     FWCore/SOA
// Class  :     ColumnValues
// 
/**\class ColumnValues ColumnValues.h "ColumnValues.h"

 Description: Provides container like access to a column of a Table

 Usage:

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 24 Aug 2017 18:13:38 GMT
//

// system include files

// user include files

// forward declarations

namespace edm {
namespace soa {

  
template<typename T>
class ColumnValues {
public:
  ColumnValues(T const* iBegin, size_t iSize):
  m_begin(iBegin), m_end(iBegin+iSize) {}
  
  T const* begin() const { return m_begin; }
  T const* end() const { return m_end; }
  
private:
  T const* m_begin = nullptr;
  T const* m_end = nullptr;
};

template<typename T>
class MutableColumnValues {
public:
  MutableColumnValues(T* iBegin, size_t iSize):
  m_begin(iBegin), m_end(iBegin+iSize) {}
  
  T* begin() const { return m_begin; }
  T* end() const { return m_end; }

private:
  T* m_begin = nullptr;
  T* m_end = nullptr;
};

}
}

#endif
