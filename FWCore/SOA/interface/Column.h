#ifndef FWCore_SOA_Column_h
#define FWCore_SOA_Column_h
// -*- C++ -*-
//
// Package:     FWCore/SOA
// Class  :     Column
// 
/**\class Column Column.h "Column.h"

 Description: Column describes a column in a Table

 Usage:
    Instances of the Column class are not intended to be used.
 Instead, the specific Column type is used as a template argument
 to a edm::soa::Table<> to describe a column in the table.
 
 A column is defined by a name and a C++ type.

 When declaring a Column template instantiation, the name
 should be declared as a constexpr const char []
 
 \code
 namespace edm {
 namespace soa {
   constexpr const char kEta[] = "eta";
   using Eta = Column<kEta,double>;
 }
 }
 \endcode

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 24 Aug 2017 16:17:56 GMT
//

// system include files

// user include files

// forward declarations
namespace edm {
namespace soa {

/**Helper class used to fill a column of a table
  in a 'non standard' way
*/
template<typename COL, typename F>
struct ColumnFillerHolder {
  using Column_type = COL;
  F m_f;
};  

template <const char* LABEL, typename T>
struct Column
{
  using type = T;
  static constexpr char const * const kLabel = LABEL;
  
  static const char* const& label() {
    static char const* const s_label(LABEL);
    return s_label;
  }
  
  template <typename F>
  static ColumnFillerHolder<Column<LABEL,T>,F> filler(F&& iF) { return {iF}; }
  
 private:
  Column() = default;
  Column(const Column&) = delete; // stop default
  
  const Column& operator=(const Column&) = delete; // stop default
};

}
}
#endif
