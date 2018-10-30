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
    Class instances inheriting from the Column class are not intended to be used.
 Instead, the specific Column type is used as a template argument
 to a edm::soa::Table<> to describe a column in the table.
 
 A column is defined by a name and a C++ type.

Classes inheriting from Column must declare a 'constexpr const char' array named 'kLabel'. 
 \code
 namespace edm {
 namespace soa {
   struct Eta : public Column<double,Eta> {
     static constexpr const char * const kLabel = "eta";
   };
 }
 }
 \endcode
Alternatively, one can use the macro SOA_DECLARE_COLUMN
\code
namespace edm {
namespace soa {

SOA_DECLARE_COLUMN(Eta,double, "eta");

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

template <typename T, typename INHERIT>
struct Column
{
  using type = T;
  
  static constexpr const char* const& label() {
    return INHERIT::kLabel;
  }
  
  template <typename F>
  static ColumnFillerHolder<INHERIT,F> filler(F&& iF) { return {iF}; }
  
 private:
  Column() = default;
  Column(const Column&) = delete; // stop default
  
  const Column& operator=(const Column&) = delete; // stop default
};

}
}
#define SOA_DECLARE_COLUMN(_ClassName_,_Type_,_String_) \
  struct _ClassName_ : public edm::soa::Column<_Type_,_ClassName_> {static constexpr const char * const kLabel=_String_; }
#endif
