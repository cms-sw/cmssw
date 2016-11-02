#ifndef FWCore_Utilities_interface_Column_h
#define FWCore_Utilities_interface_Column_h

// -*- C++ -*-
//
// Package:     Utilities
// Class  :     Column
//
/**\class Column Column.h "FWCore/Utilities/interface/Column.h"

 Description: Helper/IO manipulator for forming columns used in tabular output.

 Usage:

 A Column object can be defined in two ways:
 \code
 edm::Column col1 {"FirstNames"}; // Default width is size of the string "FirstNames"
 edm::Column col2 {"LastNames", 30}; // Width is explicitly specifed as 30.
 \endcode

 After created the column objects we, one can use them with anything
 that supports insertion operations (e.g.):

 \code
 std::cout << col1 << col2 << '\n'; // Print column titles with the widths associated for col1 and col2.
 for (auto const& name : names)
   std::cout << col1(name.first()) << col2(name.last()) << '\n';
 \endcode

 The values 'name.first()' and 'name.last()' are printed into a column
 with the width associated with col1 and col2, respectively.

*/
//
// Original Author:  Kyle Knoepfel
//         Created:
// $Id$
//

#include <iomanip>
#include <string>

namespace edm {

  class Column;

  template <typename T>
  struct ColumnEntry {
    Column const& col;
    T t;
  };

  class Column {
  public:
    explicit Column(std::string const& t);
    explicit Column(std::string const& t, std::size_t const w);

    template <typename T>
    auto operator()(T const& t) const
    {
      return ColumnEntry<T>{*this, t};
    }

  private:
    std::string title_;
    std::size_t width_;

    template <typename T>
    friend T& operator<<(T&, Column const&);

    template <typename T, typename E>
    friend T& operator<<(T&, ColumnEntry<E> const&);
  };

  template <typename T>
  T& operator<<(T& t, Column const& c)
  {
    t << std::setw(c.width_) << c.title_;
    return t;
  }

  template <typename T, typename E>
  T& operator<<(T& t, ColumnEntry<E> const& ce)
  {
    t << std::setw(ce.col.width_) << ce.t;
    return t;
  }

}

#endif
