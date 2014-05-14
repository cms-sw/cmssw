// file      : xsd/cxx/tree/serialization.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_SERIALIZATION_HXX
#define XSD_CXX_TREE_SERIALIZATION_HXX

#include <sstream>

#include <xercesc/dom/DOMElement.hpp>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      //
      //
      template <typename C>
      class list_stream
      {
      public:
        list_stream (std::basic_ostringstream<C>& os,
                     xercesc::DOMElement& parent)
            : os_ (os), parent_ (parent)
        {
        }

        std::basic_ostringstream<C>& os_;
        xercesc::DOMElement& parent_;
      };

      template <typename T>
      class as_double
      {
      public:
        as_double (const T& v)
            : x (v)
        {
        }

        const T& x;
      };

      template <typename T>
      class as_decimal
      {
      public:
        as_decimal (const T& v, const facet* f = 0)
            : x (v), facets (f)
        {
        }

        const T& x;
        const facet* facets;
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/serialization.txx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/serialization/date-time.txx>

#endif  // XSD_CXX_TREE_SERIALIZATION_HXX
