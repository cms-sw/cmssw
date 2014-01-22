// file      : xsd/cxx/tree/serialization/unsigned-int.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_SERIALIZATION_UNSIGNED_INT_HXX
#define XSD_CXX_TREE_SERIALIZATION_UNSIGNED_INT_HXX

#include <sstream>

namespace XERCES_CPP_NAMESPACE
{
  inline void
  operator<< (xercesc::DOMElement& e, unsigned int i)
  {
    std::basic_ostringstream<char> os;
    os << i;
    e << os.str ();
  }

  inline void
  operator<< (xercesc::DOMAttr& a, unsigned int i)
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
      operator<< (list_stream<C>& ls, unsigned int i)
      {
        ls.os_ << i;
      }
    }
  }
}

#endif // XSD_CXX_TREE_SERIALIZATION_UNSIGNED_INT_HXX
