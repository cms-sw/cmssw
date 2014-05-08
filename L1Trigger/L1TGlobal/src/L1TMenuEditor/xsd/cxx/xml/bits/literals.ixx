// file      : xsd/cxx/xml/bits/literals.ixx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_BITS_LITERALS_IXX
#define XSD_CXX_XML_BITS_LITERALS_IXX

#endif // XSD_CXX_XML_BITS_LITERALS_IXX


#if defined(XSD_USE_CHAR) || !defined(XSD_USE_WCHAR)

#ifndef XSD_CXX_XML_BITS_LITERALS_IXX_CHAR
#define XSD_CXX_XML_BITS_LITERALS_IXX_CHAR

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace bits
      {
        template <>
        inline const char*
        xml_prefix<char> ()
        {
          return "xml";
        }

        template <>
        inline const char*
        xml_namespace<char> ()
        {
          return "http://www.w3.org/XML/1998/namespace";
        }

        template <>
        inline const char*
        xmlns_prefix<char> ()
        {
          return "xmlns";
        }

        template <>
        inline const char*
        xmlns_namespace<char> ()
        {
          return "http://www.w3.org/2000/xmlns/";
        }

        template <>
        inline const char*
        xsi_prefix<char> ()
        {
          return "xsi";
        }

        template <>
        inline const char*
        xsi_namespace<char> ()
        {
          return "http://www.w3.org/2001/XMLSchema-instance";
        }

        template <>
        inline const char*
        type<char> ()
        {
          return "type";
        }

        template <>
        inline const char*
        nil_lit<char> ()
        {
          return "nil";
        }

        template <>
        inline const char*
        schema_location<char> ()
        {
          return "schemaLocation";
        }

        template <>
        inline const char*
        no_namespace_schema_location<char> ()
        {
          return "noNamespaceSchemaLocation";
        }

        template <>
        inline const char*
        first_prefix<char> ()
        {
          return "p1";
        }

        template <>
        inline const char*
        second_prefix<char> ()
        {
          return "p2";
        }

        template <>
        inline const char*
        third_prefix<char> ()
        {
          return "p3";
        }

        template <>
        inline const char*
        fourth_prefix<char> ()
        {
          return "p4";
        }

        template <>
        inline const char*
        fifth_prefix<char> ()
        {
          return "p5";
        }
      }
    }
  }
}

#endif // XSD_CXX_XML_BITS_LITERALS_IXX_CHAR
#endif // XSD_USE_CHAR


#if defined(XSD_USE_WCHAR) || !defined(XSD_USE_CHAR)

#ifndef XSD_CXX_XML_BITS_LITERALS_IXX_WCHAR
#define XSD_CXX_XML_BITS_LITERALS_IXX_WCHAR

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace bits
      {
        template <>
        inline const wchar_t*
        xml_prefix<wchar_t> ()
        {
          return L"xml";
        }

        template <>
        inline const wchar_t*
        xml_namespace<wchar_t> ()
        {
          return L"http://www.w3.org/XML/1998/namespace";
        }

        template <>
        inline const wchar_t*
        xmlns_prefix<wchar_t> ()
        {
          return L"xmlns";
        }

        template <>
        inline const wchar_t*
        xmlns_namespace<wchar_t> ()
        {
          return L"http://www.w3.org/2000/xmlns/";
        }

        template <>
        inline const wchar_t*
        xsi_prefix<wchar_t> ()
        {
          return L"xsi";
        }

        template <>
        inline const wchar_t*
        xsi_namespace<wchar_t> ()
        {
          return L"http://www.w3.org/2001/XMLSchema-instance";
        }

        template <>
        inline const wchar_t*
        type<wchar_t> ()
        {
          return L"type";
        }

        template <>
        inline const wchar_t*
        nil_lit<wchar_t> ()
        {
          return L"nil";
        }

        template <>
        inline const wchar_t*
        schema_location<wchar_t> ()
        {
          return L"schemaLocation";
        }

        template <>
        inline const wchar_t*
        no_namespace_schema_location<wchar_t> ()
        {
          return L"noNamespaceSchemaLocation";
        }

        template <>
        inline const wchar_t*
        first_prefix<wchar_t> ()
        {
          return L"p1";
        }

        template <>
        inline const wchar_t*
        second_prefix<wchar_t> ()
        {
          return L"p2";
        }

        template <>
        inline const wchar_t*
        third_prefix<wchar_t> ()
        {
          return L"p3";
        }

        template <>
        inline const wchar_t*
        fourth_prefix<wchar_t> ()
        {
          return L"p4";
        }

        template <>
        inline const wchar_t*
        fifth_prefix<wchar_t> ()
        {
          return L"p5";
        }
      }
    }
  }
}

#endif // XSD_CXX_XML_BITS_LITERALS_IXX_WCHAR
#endif // XSD_USE_WCHAR
