#ifndef FWCore_SOA_TableView_h
#define FWCore_SOA_TableView_h
// -*- C++ -*-
//
// Package:     FWCore/SOA
// Class  :     TableView
// 
/**\class TableView TableView.h "TableView.h"

 Description: A view of certain columns of a edm::soa::Table

 Usage:
    A TableView<> type can be constructed from any edm::soa::Table<>
 that shares the same template arguments as the TableView. E.g.
 \code
 Table<Eta,Phi> epTable;
 TableView<Eta,Phi> epView{epTable}; //compiles
 TableView<Phi,Eta> peView{epTable}; //compiles and works properly
 TableView<Eta> eView{epTable}; //compiles
 
 //TableVew<Delta> dView{epTable}; //does not compile
 \endcode

 TableViews are particularly useful when defining functions intended to
 operate on Tables.
 \code
 edm::soa::Table<Eta,Phi> sphericalAngles(edm::soa::TableView<X,Y,Z>);
 \endcode

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 25 Aug 2017 19:31:50 GMT
//

// system include files
#include <tuple>
#include <array>

// user include files
#include "FWCore/SOA/interface/Table.h"
#include "FWCore/SOA/interface/ColumnValues.h"

// forward declarations

namespace edm {
  namespace soa {
template <typename... Args>
class TableView {
  
public:
  using Layout = std::tuple<Args...>;
  static constexpr const size_t kNColumns = sizeof...(Args);
  using const_iterator = ConstTableItr<Args...>;
  
  template <typename... OArgs>
  TableView( Table<OArgs...> const& iTable):
  m_size(iTable.size()) {
    fillArray<0>(iTable,std::true_type{});
  }
  TableView( unsigned int iSize, std::array<void*, sizeof...(Args)>& iArray):
  m_size(iSize),
  m_values(iArray) {}
  
  TableView( unsigned int iSize, std::array<void const*, sizeof...(Args)>& iArray):
  m_size(iSize),
  m_values(iArray) {}
  
  unsigned int size() const {
    return m_size;
  }
  
  template<typename U>
  typename U::type const& get(size_t iRow) const {
    return static_cast<typename U::type*>(columnAddress<U>())+iRow;
  }
  
  template<typename U>
  ColumnValues<typename U::type> column() const {
    return ColumnValues<typename U::type>{static_cast<typename U::type const*>(columnAddress<U>()), m_size};
  }
  
  const_iterator begin() const { return const_iterator{m_values}; }
  const_iterator end() const { return const_iterator{m_values,size()}; }
  
private:
  std::array<void const*, sizeof...(Args)> m_values;
  unsigned int m_size;
  
  template<typename U>
  void const* columnAddress() const {
    return m_values[impl::GetIndex<0,U,Layout>::index];
  }
  
  template <int I, typename T>
  void fillArray( T const& iTable, std::true_type) {
    using ElementType = typename std::tuple_element<I, Layout>::type;
    m_values[I] = iTable.columnAddressWorkaround(static_cast<ElementType const*>(nullptr));
    fillArray<I+1>(iTable, std::conditional_t<I+1<sizeof...(Args), std::true_type, std::false_type>{});
  }
  template <int I, typename T>
  void fillArray( T const& iTable, std::false_type) {}
  
  
};

}
}

#endif
