// file      : xsd/cxx/parser/non-validating/xml-schema-pskel.ixx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#if defined(XSD_CXX_PARSER_USE_CHAR) || !defined(XSD_CXX_PARSER_USE_WCHAR)

#ifndef XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PSKEL_IXX_CHAR
#define XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PSKEL_IXX_CHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace non_validating
      {
        template<>
        inline const char* any_type_pskel<char>::
        _static_type ()
        {
          return "anyType http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* any_type_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* any_simple_type_pskel<char>::
        _static_type ()
        {
          return "anySimpleType http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* any_simple_type_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* boolean_pskel<char>::
        _static_type ()
        {
          return "boolean http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* boolean_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* byte_pskel<char>::
        _static_type ()
        {
          return "byte http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* byte_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* unsigned_byte_pskel<char>::
        _static_type ()
        {
          return "unsignedByte http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* unsigned_byte_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* short_pskel<char>::
        _static_type ()
        {
          return "short http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* short_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* unsigned_short_pskel<char>::
        _static_type ()
        {
          return "unsignedShort http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* unsigned_short_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* int_pskel<char>::
        _static_type ()
        {
          return "int http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* int_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* unsigned_int_pskel<char>::
        _static_type ()
        {
          return "unsignedInt http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* unsigned_int_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* long_pskel<char>::
        _static_type ()
        {
          return "long http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* long_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* unsigned_long_pskel<char>::
        _static_type ()
        {
          return "unsignedLong http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* unsigned_long_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* integer_pskel<char>::
        _static_type ()
        {
          return "integer http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* integer_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* negative_integer_pskel<char>::
        _static_type ()
        {
          return "negativeInteger http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* negative_integer_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* non_positive_integer_pskel<char>::
        _static_type ()
        {
          return "nonPositiveInteger http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* non_positive_integer_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* positive_integer_pskel<char>::
        _static_type ()
        {
          return "positiveInteger http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* positive_integer_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* non_negative_integer_pskel<char>::
        _static_type ()
        {
          return "nonNegativeInteger http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* non_negative_integer_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* float_pskel<char>::
        _static_type ()
        {
          return "float http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* float_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* double_pskel<char>::
        _static_type ()
        {
          return "double http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* double_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* decimal_pskel<char>::
        _static_type ()
        {
          return "decimal http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* decimal_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* string_pskel<char>::
        _static_type ()
        {
          return "string http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* string_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* normalized_string_pskel<char>::
        _static_type ()
        {
          return "normalizedString http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* normalized_string_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* token_pskel<char>::
        _static_type ()
        {
          return "token http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* token_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* name_pskel<char>::
        _static_type ()
        {
          return "Name http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* name_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* nmtoken_pskel<char>::
        _static_type ()
        {
          return "NMTOKEN http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* nmtoken_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* nmtokens_pskel<char>::
        _static_type ()
        {
          return "NMTOKENS http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* nmtokens_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* ncname_pskel<char>::
        _static_type ()
        {
          return "NCName http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* ncname_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* id_pskel<char>::
        _static_type ()
        {
          return "ID http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* id_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* idref_pskel<char>::
        _static_type ()
        {
          return "IDREF http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* idref_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* idrefs_pskel<char>::
        _static_type ()
        {
          return "IDREFS http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* idrefs_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* language_pskel<char>::
        _static_type ()
        {
          return "language http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* language_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* uri_pskel<char>::
        _static_type ()
        {
          return "anyURI http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* uri_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* qname_pskel<char>::
        _static_type ()
        {
          return "QName http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* qname_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* base64_binary_pskel<char>::
        _static_type ()
        {
          return "base64Binary http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* base64_binary_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* hex_binary_pskel<char>::
        _static_type ()
        {
          return "hexBinary http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* hex_binary_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* gday_pskel<char>::
        _static_type ()
        {
          return "gDay http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* gday_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* gmonth_pskel<char>::
        _static_type ()
        {
          return "gMonth http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* gmonth_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* gyear_pskel<char>::
        _static_type ()
        {
          return "gYear http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* gyear_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* gmonth_day_pskel<char>::
        _static_type ()
        {
          return "gMonthDay http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* gmonth_day_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* gyear_month_pskel<char>::
        _static_type ()
        {
          return "gYearMonth http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* gyear_month_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* date_pskel<char>::
        _static_type ()
        {
          return "date http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* date_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* time_pskel<char>::
        _static_type ()
        {
          return "time http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* time_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* date_time_pskel<char>::
        _static_type ()
        {
          return "dateTime http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* date_time_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const char* duration_pskel<char>::
        _static_type ()
        {
          return "duration http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const char* duration_pskel<char>::
        _dynamic_type () const
        {
          return _static_type ();
        }
      }
    }
  }
}

#endif // XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PSKEL_IXX_CHAR
#endif // XSD_CXX_PARSER_USE_CHAR


#if defined(XSD_CXX_PARSER_USE_WCHAR) || !defined(XSD_CXX_PARSER_USE_CHAR)

#ifndef XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PSKEL_IXX_WCHAR
#define XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PSKEL_IXX_WCHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace non_validating
      {
        template<>
        inline const wchar_t* any_type_pskel<wchar_t>::
        _static_type ()
        {
          return L"anyType http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* any_type_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* any_simple_type_pskel<wchar_t>::
        _static_type ()
        {
          return L"anySimpleType http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* any_simple_type_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* boolean_pskel<wchar_t>::
        _static_type ()
        {
          return L"boolean http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* boolean_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* byte_pskel<wchar_t>::
        _static_type ()
        {
          return L"byte http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* byte_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* unsigned_byte_pskel<wchar_t>::
        _static_type ()
        {
          return L"unsignedByte http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* unsigned_byte_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* short_pskel<wchar_t>::
        _static_type ()
        {
          return L"short http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* short_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* unsigned_short_pskel<wchar_t>::
        _static_type ()
        {
          return L"unsignedShort http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* unsigned_short_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* int_pskel<wchar_t>::
        _static_type ()
        {
          return L"int http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* int_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* unsigned_int_pskel<wchar_t>::
        _static_type ()
        {
          return L"unsignedInt http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* unsigned_int_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* long_pskel<wchar_t>::
        _static_type ()
        {
          return L"long http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* long_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* unsigned_long_pskel<wchar_t>::
        _static_type ()
        {
          return L"unsignedLong http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* unsigned_long_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* integer_pskel<wchar_t>::
        _static_type ()
        {
          return L"integer http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* integer_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* negative_integer_pskel<wchar_t>::
        _static_type ()
        {
          return L"negativeInteger http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* negative_integer_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* non_positive_integer_pskel<wchar_t>::
        _static_type ()
        {
          return L"nonPositiveInteger http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* non_positive_integer_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* positive_integer_pskel<wchar_t>::
        _static_type ()
        {
          return L"positiveInteger http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* positive_integer_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* non_negative_integer_pskel<wchar_t>::
        _static_type ()
        {
          return L"nonNegativeInteger http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* non_negative_integer_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* float_pskel<wchar_t>::
        _static_type ()
        {
          return L"float http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* float_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* double_pskel<wchar_t>::
        _static_type ()
        {
          return L"double http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* double_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* decimal_pskel<wchar_t>::
        _static_type ()
        {
          return L"decimal http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* decimal_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* string_pskel<wchar_t>::
        _static_type ()
        {
          return L"string http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* string_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* normalized_string_pskel<wchar_t>::
        _static_type ()
        {
          return L"normalizedString http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* normalized_string_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* token_pskel<wchar_t>::
        _static_type ()
        {
          return L"token http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* token_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* name_pskel<wchar_t>::
        _static_type ()
        {
          return L"Name http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* name_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* nmtoken_pskel<wchar_t>::
        _static_type ()
        {
          return L"NMTOKEN http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* nmtoken_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* nmtokens_pskel<wchar_t>::
        _static_type ()
        {
          return L"NMTOKENS http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* nmtokens_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* ncname_pskel<wchar_t>::
        _static_type ()
        {
          return L"NCName http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* ncname_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* id_pskel<wchar_t>::
        _static_type ()
        {
          return L"ID http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* id_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* idref_pskel<wchar_t>::
        _static_type ()
        {
          return L"IDREF http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* idref_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* idrefs_pskel<wchar_t>::
        _static_type ()
        {
          return L"IDREFS http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* idrefs_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* language_pskel<wchar_t>::
        _static_type ()
        {
          return L"language http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* language_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* uri_pskel<wchar_t>::
        _static_type ()
        {
          return L"anyURI http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* uri_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* qname_pskel<wchar_t>::
        _static_type ()
        {
          return L"QName http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* qname_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* base64_binary_pskel<wchar_t>::
        _static_type ()
        {
          return L"base64Binary http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* base64_binary_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* hex_binary_pskel<wchar_t>::
        _static_type ()
        {
          return L"hexBinary http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* hex_binary_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* gday_pskel<wchar_t>::
        _static_type ()
        {
          return L"gDay http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* gday_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* gmonth_pskel<wchar_t>::
        _static_type ()
        {
          return L"gMonth http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* gmonth_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* gyear_pskel<wchar_t>::
        _static_type ()
        {
          return L"gYear http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* gyear_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* gmonth_day_pskel<wchar_t>::
        _static_type ()
        {
          return L"gMonthDay http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* gmonth_day_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* gyear_month_pskel<wchar_t>::
        _static_type ()
        {
          return L"gYearMonth http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* gyear_month_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* date_pskel<wchar_t>::
        _static_type ()
        {
          return L"date http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* date_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* time_pskel<wchar_t>::
        _static_type ()
        {
          return L"time http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* time_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* date_time_pskel<wchar_t>::
        _static_type ()
        {
          return L"dateTime http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* date_time_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }

        template<>
        inline const wchar_t* duration_pskel<wchar_t>::
        _static_type ()
        {
          return L"duration http://www.w3.org/2001/XMLSchema";
        }

        template<>
        inline const wchar_t* duration_pskel<wchar_t>::
        _dynamic_type () const
        {
          return _static_type ();
        }
      }
    }
  }
}

#endif // XSD_CXX_PARSER_NON_VALIDATING_XML_SCHEMA_PSKEL_IXX_WCHAR
#endif // XSD_CXX_PARSER_USE_WCHAR
