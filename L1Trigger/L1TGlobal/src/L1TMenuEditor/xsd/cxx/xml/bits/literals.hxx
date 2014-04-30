// file      : xsd/cxx/xml/bits/literals.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_BITS_LITERALS_HXX
#define XSD_CXX_XML_BITS_LITERALS_HXX

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace bits
      {
        template <typename C>
        const C*
        xml_prefix ();

        template <typename C>
        const C*
        xml_namespace ();

        template <typename C>
        const C*
        xmlns_prefix ();

        template <typename C>
        const C*
        xmlns_namespace ();

        template <typename C>
        const C*
        xsi_prefix ();

        template <typename C>
        const C*
        xsi_namespace ();

        template <typename C>
        const C*
        type ();

        template <typename C>
        const C*
        nil_lit ();

        template <typename C>
        const C*
        schema_location ();

        template <typename C>
        const C*
        no_namespace_schema_location ();

        template <typename C>
        const C*
        first_prefix ();

        template <typename C>
        const C*
        second_prefix ();

        template <typename C>
        const C*
        third_prefix ();

        template <typename C>
        const C*
        fourth_prefix ();

        template <typename C>
        const C*
        fifth_prefix ();
      }
    }
  }
}

#endif  // XSD_CXX_XML_BITS_LITERALS_HXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/bits/literals.ixx>
