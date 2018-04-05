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

 The MutableRowView allows one to modify the values associated with the row.
 \code
 MutableRowView<Eta,Phi> rv = table.row(3);
 
 rv.get<Eta>() = 4;
 rv.set<Eta>(5).set<Phi>(6);
 \endcode
 
 If the necessary fillers (See ColumnFillers.h) have been defined, then
 one can directly copy values from an object into the row elements
 \code
 MutableRowView<Eta,Phi> rv = table.row(3);
 ...
 rv.copyValuesFrom( Angles{0.2,3.1415} );
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
#include "FWCore/SOA/interface/ColumnFillers.h"

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
  typename U::type& get()  {
    return *(static_cast<typename U::type*>(columnAddress<U>()));
  }
  template<typename U>
  typename U::type const& get() const  {
    return *(static_cast<typename U::type const*>(columnAddress<U>()));
  }

  template<typename U>
  MutableRowView<Args...>& set( typename U::type const& iValue)  {
    get<U>() = iValue;
    return *this;
  }
  
  template<typename U>
  void * columnAddress()  {
    return m_values[impl::GetIndex<0,U,Layout>::index];
  }
  template<typename U>
  void const * columnAddress() const {
    return m_values[impl::GetIndex<0,U,Layout>::index];
  }

  
  template<typename O>
  void copyValuesFrom(O const& iObj) {
    copyValueFromImpl<0>(iObj, std::true_type{});
  }
  template<typename O, typename... CArgs>
  void copyValuesFrom(O const& iObj, ColumnFillers<CArgs...> iFiller) {
    copyValuesUsingFiller<0>(iFiller, iObj, m_values, std::true_type{});
  }
  
private:
  template<int I, typename O>
  void copyValueFromImpl(O const& iObj, std::true_type) {
    using ColumnType = typename std::tuple_element<I,Layout>::type;
    using Type = typename ColumnType::type;
    auto ptr = static_cast<Type*>(m_values[I]);
    *ptr =value_for_column(iObj, static_cast<ColumnType*>(nullptr));
    copyValueFromImpl<I+1>(iObj, std::conditional_t<I+1 == sizeof...(Args), std::false_type, std::true_type>{});
  }
  template<int I, typename O>
  void copyValueFromImpl(O const& iObj, std::false_type) {
  }
  
  template<int I, typename E, typename F>
  static void copyValuesUsingFiller(F& iFiller, E const& iItem, std::array<void *, sizeof...(Args)>& oValues, std::true_type) {
    using Layout = std::tuple<Args...>;
    using ColumnType = typename std::tuple_element<I,Layout>::type;
    using Type = typename ColumnType::type;
    Type* pElement = static_cast<Type*>(oValues[I]);
    *pElement = iFiller.value(iItem, static_cast<ColumnType*>(nullptr));
    copyValuesUsingFiller<I+1>(iFiller,iItem, oValues, std::conditional_t<I+1==sizeof...(Args),
                                std::false_type,
                                std::true_type>{});
  }
  template<int I, typename E, typename F>
  static void copyValuesUsingFiller(F&, E const& , std::array<void *, sizeof...(Args)>& oValues,  std::false_type) {}

};

}
}


#endif
