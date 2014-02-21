// file      : xsd/cxx/tree/serialization/boolean.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_SERIALIZATION_BOOLEAN_HXX
#define XSD_CXX_TREE_SERIALIZATION_BOOLEAN_HXX

#include <sstream>

namespace XERCES_CPP_NAMESPACE
{
  inline void
  operator<< (xercesc::DOMElement& e, bool b)
  {
    std::basic_ostringstream<char> os;
    os.setf (std::ios_base::boolalpha);
    os << b;
    e << os.str ();
  }

  inline void
  operator<< (xercesc::DOMAttr& a, bool b)
  {
    std::basic_ostringstream<char> os;
    os.setf (std::ios_base::boolalpha);
    os << b;
    a << os.str ();
  }
}

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      template <typename C>
      inline void
      operator<< (list_stream<C>& ls, bool b)
      {
        // We don't need to restore the original bool format flag
        // since items in the list are all of the same type.
        //
        ls.os_.setf (std::ios_base::boolalpha);
        ls.os_ << b;
      }
    }
  }
}

#endif // XSD_CXX_TREE_SERIALIZATION_BOOLEAN_HXX
