// file      : xsd/cxx/tree/serialization/long.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_SERIALIZATION_LONG_HXX
#define XSD_CXX_TREE_SERIALIZATION_LONG_HXX

#include <sstream>

namespace XERCES_CPP_NAMESPACE
{
  inline void
  operator<< (xercesc::DOMElement& e, long long l)
  {
    std::basic_ostringstream<char> os;
    os << l;
    e << os.str ();
  }

  inline void
  operator<< (xercesc::DOMAttr& a, long long l)
  {
    std::basic_ostringstream<char> os;
    os << l;
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
      operator<< (list_stream<C>& ls, long long l)
      {
        ls.os_ << l;
      }
    }
  }
}

#endif // XSD_CXX_TREE_SERIALIZATION_LONG_HXX
