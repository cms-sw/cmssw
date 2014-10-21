// file      : xsd/cxx/parser/non-validating/xml-schema-pimpl.ixx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#if defined(XSD_CXX_PARSER_USE_CHAR) || !defined(XSD_CXX_PARSER_USE_WCHAR)

#ifndef XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PIMPL_IXX_CHAR
#define XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PIMPL_IXX_CHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace non_validating
      {
        namespace bits
        {
          //
          //
          template<>
          inline const char*
          positive_inf<char> ()
          {
            return "INF";
          }

          template<>
          inline const char*
          negative_inf<char> ()
          {
            return "-INF";
          }

          template<>
          inline const char*
          nan<char> ()
          {
            return "NaN";
          }

          //
          //
          template<>
          inline const char*
          true_<char> ()
          {
            return "true";
          }

          template<>
          inline const char*
          one<char> ()
          {
            return "1";
          }
        }
      }
    }
  }
}

#endif // XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PIMPL_IXX_CHAR
#endif // XSD_CXX_PARSER_USE_CHAR


#if defined(XSD_CXX_PARSER_USE_WCHAR) || !defined(XSD_CXX_PARSER_USE_CHAR)

#ifndef XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PIMPL_IXX_WCHAR
#define XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PIMPL_IXX_WCHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace non_validating
      {
        namespace bits
        {
          //
          //
          template<>
          inline const wchar_t*
          positive_inf<wchar_t> ()
          {
            return L"INF";
          }

          template<>
          inline const wchar_t*
          negative_inf<wchar_t> ()
          {
            return L"-INF";
          }

          template<>
          inline const wchar_t*
          nan<wchar_t> ()
          {
            return L"NaN";
          }

          //
          //
          template<>
          inline const wchar_t*
          true_<wchar_t> ()
          {
            return L"true";
          }

          template<>
          inline const wchar_t*
          one<wchar_t> ()
          {
            return L"1";
          }
        }
      }
    }
  }
}

#endif // XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PIMPL_IXX_WCHAR
#endif // XSD_CXX_PARSER_USE_WCHAR
