#ifndef FWCore_SOA_TableExaminer_h
#define FWCore_SOA_TableExaminer_h
// -*- C++ -*-
//
// Package:     FWCore/SOA
// Class  :     TableExaminer
// 
/**\class TableExaminer TableExaminer.h "TableExaminer.h"

 Description: Concrete implementation of TableExaminerBase

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon, 28 Aug 2017 14:22:29 GMT
//

// system include files

// user include files
#include "FWCore/SOA/interface/TableExaminerBase.h"
#include "FWCore/SOA/interface/Table.h"

// forward declarations
namespace edm {
namespace soa {

template <typename T>
class TableExaminer : public TableExaminerBase
{

   public:
  explicit TableExaminer( T const* iTable):
  m_table (iTable) {}
  
  TableExaminer(const TableExaminer<T>&) = default;
  
  TableExaminer<T>& operator=(const TableExaminer<T>&) = default;

  ~TableExaminer() override {}

  // ---------- const member functions ---------------------

  size_t size() const override final { return m_table->size(); }
  
  void const* columnAddress(unsigned int iColumnIndex) const override final {
    return m_table->columnAddressByIndex(iColumnIndex);
  }
  
  std::vector<std::type_index> columnTypes() const override final {
    std::vector<std::type_index> returnValue;
    returnValue.reserve(T::kNColumns);
    using Layout = typename T::Layout;
    columnTypesImpl<0, T::kNColumns>(returnValue, static_cast<std::true_type*>(nullptr));
    return returnValue;
  }
  
  std::vector<std::pair<char const*, std::type_index>>
  columnDescriptions() const override final {
    std::vector<std::pair<char const*, std::type_index>>  returnValue;
    returnValue.reserve(T::kNColumns);
    using Layout = typename T::Layout;
    columnDescImpl<0, T::kNColumns>(returnValue, static_cast<std::true_type*>(nullptr));
    return returnValue;
  }
  
  const std::type_info* typeID() const override final {
    return &typeid(T);
  }
  
private:
  template <int I, int S>
  void columnTypesImpl(std::vector<std::type_index>& iV, std::true_type*) const {
    using Layout = typename T::Layout;
    iV.emplace_back( typeid( typename std::tuple_element<I,Layout>::type ) );
    columnTypesImpl<I+1, S>(iV,
                            static_cast<typename std::conditional< I+1 != S,
                            std::true_type,
                            std::false_type>::type*>(nullptr));
  }
  
  template <int I, int S>
  void columnTypesImpl(std::vector<std::type_index>&, std::false_type*) const {};
  
  template <int I, int S>
  void columnDescImpl(std::vector<std::pair<char const*, std::type_index>>& iV,
                      std::true_type*) const {
    using Layout = typename T::Layout;
    using ColumnType = typename std::tuple_element<I,Layout>::type;
    iV.emplace_back( ColumnType::label(), typeid( typename ColumnType::type ) );
    columnDescImpl<I+1, S>(iV,
                           static_cast<typename std::conditional< I+1 != S,
                           std::true_type,
                           std::false_type>::type*>(nullptr));
  }
  
  template <int I, int S>
  void columnDescImpl(std::vector<std::pair<char const*, std::type_index>>&,
                      std::false_type*) const {};

  // ---------- member data --------------------------------
  T const* m_table;

};

}
}

#endif
