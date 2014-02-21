// file      : xsd/cxx/tree/serialization/unsigned-byte.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_SERIALIZATION_UNSIGNED_BYTE_HXX
#define XSD_CXX_TREE_SERIALIZATION_UNSIGNED_BYTE_HXX

#include <sstream>

namespace XERCES_CPP_NAMESPACE
{
  inline void
  operator<< (xercesc::DOMElement& e, unsigned char c)
  {
    std::basic_ostringstream<char> os;
    os << static_cast<unsigned short> (c);
    e << os.str ();
  }

  inline void
  operator<< (xercesc::DOMAttr& a, unsigned char c)
  {
    std::basic_ostringstream<char> os;
    os << static_cast<unsigned short> (c);
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
      operator<< (list_stream<C>& ls, unsigned char c)
      {
        ls.os_ << static_cast<unsigned short> (c);
      }
    }
  }
}

#endif // XSD_CXX_TREE_SERIALIZATION_UNSIGNED_BYTE_HXX
