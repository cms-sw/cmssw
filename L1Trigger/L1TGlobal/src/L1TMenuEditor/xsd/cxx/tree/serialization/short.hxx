// file      : xsd/cxx/tree/serialization/short.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_SERIALIZATION_SHORT_HXX
#define XSD_CXX_TREE_SERIALIZATION_SHORT_HXX

#include <sstream>

namespace XERCES_CPP_NAMESPACE
{
  inline void
  operator<< (xercesc::DOMElement& e, short s)
  {
    std::basic_ostringstream<char> os;
    os << s;
    e << os.str ();
  }

  inline void
  operator<< (xercesc::DOMAttr& a, short s)
  {
    std::basic_ostringstream<char> os;
    os << s;
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
      operator<< (list_stream<C>& ls, short s)
      {
        ls.os_ << s;
      }
    }
  }
}

#endif // XSD_CXX_TREE_SERIALIZATION_SHORT_HXX
