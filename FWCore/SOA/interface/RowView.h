#ifndef FWCore_SOA_RowView_h
#define FWCore_SOA_RowView_h
// -*- C++ -*-
//
// Package:     FWCore/SOA
// Class  :     RowView
// 
/**\class RowView RowView.h "RowView.h"

 Description: Provides access to all elements of a row in a edm::soa::Table

 Usage:
    Individual column entries of a row are accessed via the 'get' method
 by specifying the edm::soa::Column type for the column.
 \code
 RowView<Eta,Phi> rv{...};
 
 auto eta = rv.get<Eta>();
 \endcode

*/
//
// Original Author:  Chris Jones
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
