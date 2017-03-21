#ifndef FWCore_Utilities_interface_OStreamColumn_h
#define FWCore_Utilities_interface_OStreamColumn_h

// -*- C++ -*-
//
// Package:     Utilities
// Class  :     OStreamColumn
//
/**\class OStreamColumn OStreamColumn.h "FWCore/Utilities/interface/OStreamColumn.h"

 Description: Helper/IO manipulator for forming columns used in tabular output.

 Usage:

 An OStreamColumn object can be defined in two ways:
 \code
 edm::OStreamColumn col1 {"FirstNames"}; // Default width is size of the string "FirstNames"
 edm::OStreamColumn col2 {"LastNames", 30}; // Width is explicitly specifed as 30.
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

  class OStreamColumn;

  template <typename T>
  struct OStreamColumnEntry {
    OStreamColumn const& col;
    T t;
  };

  class OStreamColumn {
  public:
    explicit OStreamColumn(std::string const& t);
    explicit OStreamColumn(std::string const& t, std::size_t const w);

    template <typename T>
    auto operator()(T const& t) const
    {
      return OStreamColumnEntry<T>{*this, t};
    }

    std::size_t width() const { return width_; }

  private:
    std::string title_;
    std::size_t width_;

    friend std::ostream& operator<<(std::ostream&, OStreamColumn const&);

    template <typename E>
    friend std::ostream& operator<<(std::ostream&, OStreamColumnEntry<E> const&);
  };

  std::ostream& operator<<(std::ostream& t, OStreamColumn const& c);

  template <typename E>
  std::ostream& operator<<(std::ostream& t, OStreamColumnEntry<E> const& ce)
  {
    t << std::setw(ce.col.width_) << ce.t;
    return t;
  }

}

#endif
