// file      : xsd/cxx/parser/validating/exceptions.ixx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#if defined(XSD_CXX_PARSER_USE_CHAR) || !defined(XSD_CXX_PARSER_USE_WCHAR)

#ifndef XSD_CXX_PARSER_VALIDATING_EXCEPTIONS_IXX_CHAR
#define XSD_CXX_PARSER_VALIDATING_EXCEPTIONS_IXX_CHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {
        // expected_attribute
        //
        template<>
        inline
        std::basic_string<char> expected_attribute<char>::
        message () const
        {
          std::basic_string<char> r ("expected attribute '");
          r += expected_namespace_;
          r += expected_namespace_.empty () ? "" : "#";
          r += expected_name_;
          r += "'";
          return r;
        }

        // unexpected_attribute
        //
        template<>
        inline
        std::basic_string<char> unexpected_attribute<char>::
        message () const
        {
          std::basic_string<char> r ("unexpected attribute '");
          r += encountered_namespace_;
          r += encountered_namespace_.empty () ? "" : "#";
          r += encountered_name_;
          r += "'";
          return r;
        }

        // unexpected_characters
        //
        template<>
        inline
        std::basic_string<char> unexpected_characters<char>::
        message () const
        {
          std::basic_string<char> r ("unexpected characters '");
          r += characters_;
          r += "'";
          return r;
        }

        // invalid_value
        //
        template<>
        inline
        std::basic_string<char> invalid_value<char>::
        message () const
        {
          std::basic_string<char> r ("'");
          r += value_;
          r += "' is not a valid value representation ";
          r += "for type '";
          r += type_;
          r += "'";
          return r;
        }
      }
    }
  }
}

#endif // XSD_CXX_PARSER_VALIDATING_EXCEPTIONS_IXX_CHAR
#endif // XSD_CXX_PARSER_USE_CHAR


#if defined(XSD_CXX_PARSER_USE_WCHAR) || !defined(XSD_CXX_PARSER_USE_CHAR)

#ifndef XSD_CXX_PARSER_VALIDATING_EXCEPTIONS_IXX_WCHAR
#define XSD_CXX_PARSER_VALIDATING_EXCEPTIONS_IXX_WCHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {
        // expected_attribute
        //
        template<>
        inline
        std::basic_string<wchar_t> expected_attribute<wchar_t>::
        message () const
        {
          std::basic_string<wchar_t> r (L"expected attribute '");
          r += expected_namespace_;
          r += expected_namespace_.empty () ? L"" : L"#";
          r += expected_name_;
          r += L"'";
          return r;
        }

        // unexpected_attribute
        //
        template<>
        inline
        std::basic_string<wchar_t> unexpected_attribute<wchar_t>::
        message () const
        {
          std::basic_string<wchar_t> r (L"unexpected attribute '");
          r += encountered_namespace_;
          r += encountered_namespace_.empty () ? L"" : L"#";
          r += encountered_name_;
          r += L"'";
          return r;
        }

        // unexpected_characters
        //
        template<>
        inline
        std::basic_string<wchar_t> unexpected_characters<wchar_t>::
        message () const
        {
          std::basic_string<wchar_t> r (L"unexpected characters '");
          r += characters_;
          r += L"'";
          return r;
        }

        // invalid_value
        //
        template<>
        inline
        std::basic_string<wchar_t> invalid_value<wchar_t>::
        message () const
        {
          std::basic_string<wchar_t> r (L"'");
          r += value_;
          r += L"' is not a valid value representation ";
          r += L"for type '";
          r += type_;
          r += L"'";
          return r;
        }
      }
    }
  }
}

#endif // XSD_CXX_PARSER_VALIDATING_EXCEPTIONS_IXX_WCHAR
#endif // XSD_CXX_PARSER_USE_WCHAR
