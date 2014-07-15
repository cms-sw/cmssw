// file      : xsd/cxx/tree/serialization/int.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_SERIALIZATION_INT_HXX
#define XSD_CXX_TREE_SERIALIZATION_INT_HXX

#include <sstream>

namespace XERCES_CPP_NAMESPACE
{
  inline void
  operator<< (xercesc::DOMElement& e, int i)
  {
    std::basic_ostringstream<char> os;
    os << i;
    e << os.str ();
  }

  inline void
  operator<< (xercesc::DOMAttr& a, int i)
  {
    std::basic_ostringstream<char> os;
    os << i;
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
      operator<< (list_stream<C>& ls, int i)
      {
        ls.os_ << i;
      }
    }
  }
}

#endif // XSD_CXX_TREE_SERIALIZATION_INT_HXX
