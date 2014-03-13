// file      : xsd/cxx/parser/schema-exceptions.ixx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#if defined(XSD_CXX_PARSER_USE_CHAR) || !defined(XSD_CXX_PARSER_USE_WCHAR)

#ifndef XSD_CXX_PARSER_SCHEMA_EXCEPTIONS_IXX_CHAR
#define XSD_CXX_PARSER_SCHEMA_EXCEPTIONS_IXX_CHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // expected_element
      //
      template<>
      inline
      std::basic_string<char> expected_element<char>::
      message () const
      {
        std::basic_string<char> r ("expected element '");
        r += expected_namespace_;
        r += expected_namespace_.empty () ? "" : "#";
        r += expected_name_;
        r += "'";

        if (!encountered_name_.empty ())
        {
          r += " instead of '";
          r +=  encountered_namespace_;
          r +=  encountered_namespace_.empty () ? "" : "#";
          r +=  encountered_name_;
          r += "'";
        }

        return r;
      }

      // unexpected_element
      //
      template<>
      inline
      std::basic_string<char> unexpected_element<char>::
      message () const
      {
        std::basic_string<char> r ("unexpected element '");
        r += encountered_namespace_;
        r += encountered_namespace_.empty () ? "" : "#";
        r += encountered_name_;
        r += "'";
        return r;
      }

      // dynamic_type
      //
      template<>
      inline
      std::basic_string<char> dynamic_type<char>::
      message () const
      {
        std::basic_string<char> r ("invalid xsi:type '");
        r += type_;
        r += "'";
        return r;
      }
    }
  }
}

#endif // XSD_CXX_PARSER_SCHEMA_EXCEPTIONS_IXX_CHAR
#endif // XSD_CXX_PARSER_USE_CHAR


#if defined(XSD_CXX_PARSER_USE_WCHAR) || !defined(XSD_CXX_PARSER_USE_CHAR)

#ifndef XSD_CXX_PARSER_SCHEMA_EXCEPTIONS_IXX_WCHAR
#define XSD_CXX_PARSER_SCHEMA_EXCEPTIONS_IXX_WCHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // expected_element
      //
      template<>
      inline
      std::basic_string<wchar_t> expected_element<wchar_t>::
      message () const
      {
        std::basic_string<wchar_t> r (L"expected element '");
        r += expected_namespace_;
        r += expected_namespace_.empty () ? L"" : L"#";
        r += expected_name_;
        r += L"'";

        if (!encountered_name_.empty ())
        {
          r += L" instead of '";
          r +=  encountered_namespace_;
          r +=  encountered_namespace_.empty () ? L"" : L"#";
          r +=  encountered_name_;
          r += L"'";
        }

        return r;
      }

      // unexpected_element
      //
      template<>
      inline
      std::basic_string<wchar_t> unexpected_element<wchar_t>::
      message () const
      {
        std::basic_string<wchar_t> r (L"unexpected element '");
        r += encountered_namespace_;
        r += encountered_namespace_.empty () ? L"" : L"#";
        r += encountered_name_;
        r += L"'";
        return r;
      }

      // dynamic_type
      //
      template<>
      inline
      std::basic_string<wchar_t> dynamic_type<wchar_t>::
      message () const
      {
        std::basic_string<wchar_t> r (L"invalid xsi:type '");
        r += type_;
        r += L"'";
        return r;
      }
    }
  }
}

#endif // XSD_CXX_PARSER_SCHEMA_EXCEPTIONS_IXX_WCHAR
#endif // XSD_CXX_PARSER_USE_WCHAR
