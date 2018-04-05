#ifndef FWCore_SOA_TableExaminerBase_h
#define FWCore_SOA_TableExaminerBase_h
// -*- C++ -*-
//
// Package:     FWCore/SOA
// Class  :     TableExaminerBase
// 
/**\class TableExaminerBase TableExaminerBase.h "TableExaminerBase.h"

 Description: Base class interface for examining a edm::soa::Table

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon, 28 Aug 2017 14:22:26 GMT
//

// system include files
#include <vector>
#include <utility>
#include <typeinfo>
#include <typeindex>

// user include files

// forward declarations

namespace edm {
namespace soa {
  
class TableExaminerBase
{

public:
  TableExaminerBase() = default;
  virtual ~TableExaminerBase() =default;
  TableExaminerBase(const TableExaminerBase&) = default;
  TableExaminerBase& operator=(const TableExaminerBase&) = default;

      // ---------- const member functions ---------------------
  virtual std::vector<std::type_index> columnTypes() const = 0;
  
  virtual std::vector<std::pair<char const*, std::type_index>> columnDescriptions() const = 0;
  
  virtual size_t size() const = 0;
  
  virtual void const* columnAddress(unsigned int iColumnIndex) const = 0;
  
  virtual const std::type_info* typeID() const = 0;

   private:

      // ---------- member data --------------------------------

};
}
}


#endif
