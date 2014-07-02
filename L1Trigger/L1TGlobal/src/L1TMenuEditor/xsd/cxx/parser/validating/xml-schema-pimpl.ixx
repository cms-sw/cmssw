// file      : xsd/cxx/parser/validating/xml-schema-pimpl.ixx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#if defined(XSD_CXX_PARSER_USE_CHAR) || !defined(XSD_CXX_PARSER_USE_WCHAR)

#ifndef XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PIMPL_IXX_CHAR
#define XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PIMPL_IXX_CHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {
        namespace bits
        {
          template<>
          inline const char*
          boolean<char> ()
          {
            return "boolean";
          }

          template<>
          inline const char*
          byte<char> ()
          {
            return "byte";
          }

          template<>
          inline const char*
          unsigned_byte<char> ()
          {
            return "unsignedByte";
          }

          template<>
          inline const char*
          short_<char> ()
          {
            return "short";
          }

          template<>
          inline const char*
          unsigned_short<char> ()
          {
            return "unsignedShort";
          }

          template<>
          inline const char*
          int_<char> ()
          {
            return "int";
          }

          template<>
          inline const char*
          unsigned_int<char> ()
          {
            return "unsignedInt";
          }

          template<>
          inline const char*
          long_<char> ()
          {
            return "long";
          }

          template<>
          inline const char*
          unsigned_long<char> ()
          {
            return "unsignedLong";
          }

          template<>
          inline const char*
          integer<char> ()
          {
            return "integer";
          }

          template<>
          inline const char*
          negative_integer<char> ()
          {
            return "negativeInteger";
          }

          template<>
          inline const char*
          non_positive_integer<char> ()
          {
            return "nonPositiveInteger";
          }

          template<>
          inline const char*
          non_negative_integer<char> ()
          {
            return "nonNegativeInteger";
          }

          template<>
          inline const char*
          positive_integer<char> ()
          {
            return "positiveInteger";
          }

          template<>
          inline const char*
          float_<char> ()
          {
            return "float";
          }

          template<>
          inline const char*
          double_<char> ()
          {
            return "double";
          }

          template<>
          inline const char*
          decimal<char> ()
          {
            return "decimal";
          }

          template<>
          inline const char*
          name<char> ()
          {
            return "Name";
          }

          template<>
          inline const char*
          nmtoken<char> ()
          {
            return "NMTOKEN";
          }

          template<>
          inline const char*
          nmtokens<char> ()
          {
            return "NMTOKENS";
          }

          template<>
          inline const char*
          ncname<char> ()
          {
            return "NCName";
          }

          template<>
          inline const char*
          id<char> ()
          {
            return "ID";
          }

          template<>
          inline const char*
          idref<char> ()
          {
            return "IDREF";
          }

          template<>
          inline const char*
          idrefs<char> ()
          {
            return "IDREFS";
          }

          template<>
          inline const char*
          language<char> ()
          {
            return "language";
          }

          template<>
          inline const char*
          qname<char> ()
          {
            return "QName";
          }

          template<>
          inline const char*
          base64_binary<char> ()
          {
            return "base64Binary";
          }

          template<>
          inline const char*
          hex_binary<char> ()
          {
            return "hexBinary";
          }

          template<>
          inline const char*
          gday<char> ()
          {
            return "gDay";
          }

          template<>
          inline const char*
          gmonth<char> ()
          {
            return "gMonth";
          }

          template<>
          inline const char*
          gyear<char> ()
          {
            return "gYear";
          }

          template<>
          inline const char*
          gmonth_day<char> ()
          {
            return "gMonthDay";
          }

          template<>
          inline const char*
          gyear_month<char> ()
          {
            return "gYearMonth";
          }

          template<>
          inline const char*
          date<char> ()
          {
            return "date";
          }

          template<>
          inline const char*
          time<char> ()
          {
            return "time";
          }

          template<>
          inline const char*
          date_time<char> ()
          {
            return "dateTime";
          }

          template<>
          inline const char*
          duration<char> ()
          {
            return "duration";
          }

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
          false_<char> ()
          {
            return "false";
          }

          template<>
          inline const char*
          one<char> ()
          {
            return "1";
          }

          template<>
          inline const char*
          zero<char> ()
          {
            return "0";
          }
        }
      }
    }
  }
}

#endif // XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PIMPL_IXX_CHAR
#endif // XSD_CXX_PARSER_USE_CHAR


#if defined(XSD_CXX_PARSER_USE_WCHAR) || !defined(XSD_CXX_PARSER_USE_CHAR)

#ifndef XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PIMPL_IXX_WCHAR
#define XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PIMPL_IXX_WCHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {
        namespace bits
        {
          template<>
          inline const wchar_t*
          boolean<wchar_t> ()
          {
            return L"boolean";
          }

          template<>
          inline const wchar_t*
          byte<wchar_t> ()
          {
            return L"byte";
          }

          template<>
          inline const wchar_t*
          unsigned_byte<wchar_t> ()
          {
            return L"unsignedByte";
          }

          template<>
          inline const wchar_t*
          short_<wchar_t> ()
          {
            return L"short";
          }

          template<>
          inline const wchar_t*
          unsigned_short<wchar_t> ()
          {
            return L"unsignedShort";
          }

          template<>
          inline const wchar_t*
          int_<wchar_t> ()
          {
            return L"int";
          }

          template<>
          inline const wchar_t*
          unsigned_int<wchar_t> ()
          {
            return L"unsignedInt";
          }

          template<>
          inline const wchar_t*
          long_<wchar_t> ()
          {
            return L"long";
          }

          template<>
          inline const wchar_t*
          unsigned_long<wchar_t> ()
          {
            return L"unsignedLong";
          }

          template<>
          inline const wchar_t*
          integer<wchar_t> ()
          {
            return L"integer";
          }

          template<>
          inline const wchar_t*
          negative_integer<wchar_t> ()
          {
            return L"negativeInteger";
          }

          template<>
          inline const wchar_t*
          non_positive_integer<wchar_t> ()
          {
            return L"nonPositiveInteger";
          }

          template<>
          inline const wchar_t*
          non_negative_integer<wchar_t> ()
          {
            return L"nonNegativeInteger";
          }

          template<>
          inline const wchar_t*
          positive_integer<wchar_t> ()
          {
            return L"positiveInteger";
          }

          template<>
          inline const wchar_t*
          float_<wchar_t> ()
          {
            return L"float";
          }

          template<>
          inline const wchar_t*
          double_<wchar_t> ()
          {
            return L"double";
          }

          template<>
          inline const wchar_t*
          decimal<wchar_t> ()
          {
            return L"decimal";
          }

          template<>
          inline const wchar_t*
          name<wchar_t> ()
          {
            return L"Name";
          }

          template<>
          inline const wchar_t*
          nmtoken<wchar_t> ()
          {
            return L"NMTOKEN";
          }

          template<>
          inline const wchar_t*
          nmtokens<wchar_t> ()
          {
            return L"NMTOKENS";
          }

          template<>
          inline const wchar_t*
          ncname<wchar_t> ()
          {
            return L"NCName";
          }

          template<>
          inline const wchar_t*
          id<wchar_t> ()
          {
            return L"ID";
          }

          template<>
          inline const wchar_t*
          idref<wchar_t> ()
          {
            return L"IDREF";
          }

          template<>
          inline const wchar_t*
          idrefs<wchar_t> ()
          {
            return L"IDREFS";
          }

          template<>
          inline const wchar_t*
          language<wchar_t> ()
          {
            return L"language";
          }

          template<>
          inline const wchar_t*
          qname<wchar_t> ()
          {
            return L"QName";
          }

          template<>
          inline const wchar_t*
          base64_binary<wchar_t> ()
          {
            return L"base64Binary";
          }

          template<>
          inline const wchar_t*
          hex_binary<wchar_t> ()
          {
            return L"hexBinary";
          }

          template<>
          inline const wchar_t*
          gday<wchar_t> ()
          {
            return L"gDay";
          }

          template<>
          inline const wchar_t*
          gmonth<wchar_t> ()
          {
            return L"gMonth";
          }

          template<>
          inline const wchar_t*
          gyear<wchar_t> ()
          {
            return L"gYear";
          }

          template<>
          inline const wchar_t*
          gmonth_day<wchar_t> ()
          {
            return L"gMonthDay";
          }

          template<>
          inline const wchar_t*
          gyear_month<wchar_t> ()
          {
            return L"gYearMonth";
          }

          template<>
          inline const wchar_t*
          date<wchar_t> ()
          {
            return L"date";
          }

          template<>
          inline const wchar_t*
          time<wchar_t> ()
          {
            return L"time";
          }

          template<>
          inline const wchar_t*
          date_time<wchar_t> ()
          {
            return L"dateTime";
          }

          template<>
          inline const wchar_t*
          duration<wchar_t> ()
          {
            return L"duration";
          }


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
          false_<wchar_t> ()
          {
            return L"false";
          }

          template<>
          inline const wchar_t*
          one<wchar_t> ()
          {
            return L"1";
          }

          template<>
          inline const wchar_t*
          zero<wchar_t> ()
          {
            return L"0";
          }
        }
      }
    }
  }
}

#endif // XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PIMPL_IXX_WCHAR
#endif // XSD_CXX_PARSER_USE_WCHAR
