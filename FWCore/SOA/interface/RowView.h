#ifndef FWCore_SOA_RowView_h
#define FWCore_SOA_RowView_h
// -*- C++ -*-
//
// Package:     FWCore/SOA
// Class  :     RowView
// 
/**\class RowView RowView.h "RowView.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Thu, 24 Aug 2017 16:49:17 GMT
//

// system include files
#include <tuple>
#include <array>

// user include files
#include "FWCore/SOA/interface/tablehelpers.h"

// forward declarations

namespace edm {
namespace soa {
  
template <typename... Args>
class RowView {
  using Layout = std::tuple<Args...>;
  std::array<void const*, sizeof...(Args)> m_values;
  
public:
  explicit RowView( std::array<void const*, sizeof...(Args)> const& iValues):
  m_values{iValues} {}
  
  template<typename U>
  typename U::type const& get() const {
    return *(static_cast<typename U::type const*>(columnAddress<U>()));
  }
  
  template<typename U>
  void const* columnAddress() const {
    return m_values[impl::GetIndex<0,U,Layout>::index];
  }
  
};

template <typename... Args>
class MutableRowView {
  using Layout = std::tuple<Args...>;
  std::array<void*, sizeof...(Args)> m_values;
  
public:
  explicit MutableRowView( std::array<void*, sizeof...(Args)>& iValues):
  m_values{iValues} {}
  
  template<typename U>
  typename U::type& get() const {
    return *(static_cast<typename U::type*>(columnAddress<U>()));
  }
  
  template<typename U>
  void * columnAddress() const {
    return m_values[impl::GetIndex<0,U,Layout>::index];
  }
  
};

}
}


#endif
