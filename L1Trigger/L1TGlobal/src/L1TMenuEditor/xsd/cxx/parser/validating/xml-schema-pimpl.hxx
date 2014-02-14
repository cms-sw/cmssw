// file      : xsd/cxx/parser/validating/xml-schema-pimpl.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PIMPL_HXX
#define XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PIMPL_HXX

#include <string>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/validating/xml-schema-pskel.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {
        // any_type
        //
        template <typename C>
        struct any_type_pimpl: virtual any_type_pskel<C>
        {
          virtual void
          post_any_type ();
        };

        // any_simple_type
        //
        template <typename C>
        struct any_simple_type_pimpl: virtual any_simple_type_pskel<C>
        {
          virtual void
          post_any_simple_type ();
        };

        // boolean
        //
        template <typename C>
        struct boolean_pimpl: virtual boolean_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual bool
          post_boolean ();

        protected:
          std::basic_string<C> str_;
          bool value_;
        };


        // 8-bit
        //
        template <typename C>
        struct byte_pimpl: virtual byte_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual signed char
          post_byte ();

        protected:
          std::basic_string<C> str_;
          signed char value_;
        };


        template <typename C>
        struct unsigned_byte_pimpl: virtual unsigned_byte_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual unsigned char
          post_unsigned_byte ();

        protected:
          std::basic_string<C> str_;
          unsigned char value_;
        };


        // 16-bit
        //
        template <typename C>
        struct short_pimpl: virtual short_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual short
          post_short ();

        protected:
          std::basic_string<C> str_;
          short value_;
        };


        template <typename C>
        struct unsigned_short_pimpl: virtual unsigned_short_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual unsigned short
          post_unsigned_short ();

        protected:
          std::basic_string<C> str_;
          unsigned short value_;
        };


        // 32-bit
        //
        template <typename C>
        struct int_pimpl: virtual int_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual int
          post_int ();

        protected:
          std::basic_string<C> str_;
          int value_;
        };


        template <typename C>
        struct unsigned_int_pimpl: virtual unsigned_int_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual unsigned int
          post_unsigned_int ();

        protected:
          std::basic_string<C> str_;
          unsigned int value_;
        };


        // 64-bit
        //
        template <typename C>
        struct long_pimpl: virtual long_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual long long
          post_long ();

        protected:
          std::basic_string<C> str_;
          long long value_;
        };


        template <typename C>
        struct unsigned_long_pimpl: virtual unsigned_long_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual unsigned long long
          post_unsigned_long ();

        protected:
          std::basic_string<C> str_;
          unsigned long long value_;
        };


        // Arbitrary-length integers.
        //
        template <typename C>
        struct integer_pimpl: virtual integer_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual long long
          post_integer ();

        protected:
          std::basic_string<C> str_;
          long long value_;
        };

        template <typename C>
        struct negative_integer_pimpl: virtual negative_integer_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual long long
          post_negative_integer ();

        protected:
          std::basic_string<C> str_;
          long long value_;
        };

        template <typename C>
        struct non_positive_integer_pimpl: virtual non_positive_integer_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual long long
          post_non_positive_integer ();

        protected:
          std::basic_string<C> str_;
          long long value_;
        };

        template <typename C>
        struct positive_integer_pimpl: virtual positive_integer_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual unsigned long long
          post_positive_integer ();

        protected:
          std::basic_string<C> str_;
          unsigned long long value_;
        };

        template <typename C>
        struct non_negative_integer_pimpl: virtual non_negative_integer_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual unsigned long long
          post_non_negative_integer ();

        protected:
          std::basic_string<C> str_;
          unsigned long long value_;
        };


        // Floats.
        //
        template <typename C>
        struct float_pimpl: virtual float_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual float
          post_float ();

        protected:
          std::basic_string<C> str_;
          float value_;
        };


        template <typename C>
        struct double_pimpl: virtual double_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual double
          post_double ();

        protected:
          std::basic_string<C> str_;
          double value_;
        };


        template <typename C>
        struct decimal_pimpl: virtual decimal_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual double
          post_decimal ();

        protected:
          std::basic_string<C> str_;
          double value_;
        };


        // Strings.
        //
        template <typename C>
        struct string_pimpl: virtual string_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual std::basic_string<C>
          post_string ();

        protected:
          std::basic_string<C> str_;
        };

        template <typename C>
        struct normalized_string_pimpl: virtual normalized_string_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual std::basic_string<C>
          post_normalized_string ();

        protected:
          std::basic_string<C> str_;
        };

        template <typename C>
        struct token_pimpl: virtual token_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual std::basic_string<C>
          post_token ();

        protected:
          std::basic_string<C> str_;
        };

        template <typename C>
        struct name_pimpl: virtual name_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual std::basic_string<C>
          post_name ();

        protected:
          std::basic_string<C> str_;
        };

        template <typename C>
        struct nmtoken_pimpl: virtual nmtoken_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual std::basic_string<C>
          post_nmtoken ();

        protected:
          std::basic_string<C> str_;
        };

        template <typename C>
        struct nmtokens_pimpl: virtual nmtokens_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _xsd_parse_item (const ro_string<C>&);

          virtual void
          _post ();

          virtual string_sequence<C>
          post_nmtokens ();

        protected:
          string_sequence<C> seq_;
          nmtoken_pimpl<C> parser_;
        };

        template <typename C>
        struct ncname_pimpl: virtual ncname_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual std::basic_string<C>
          post_ncname ();

        protected:
          std::basic_string<C> str_;
        };

        template <typename C>
        struct id_pimpl: virtual id_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual std::basic_string<C>
          post_id ();

        protected:
          std::basic_string<C> str_;
        };

        template <typename C>
        struct idref_pimpl: virtual idref_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual std::basic_string<C>
          post_idref ();

        protected:
          std::basic_string<C> str_;
        };

        template <typename C>
        struct idrefs_pimpl: virtual idrefs_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _xsd_parse_item (const ro_string<C>&);

          virtual void
          _post ();

          virtual string_sequence<C>
          post_idrefs ();

        protected:
          string_sequence<C> seq_;
          idref_pimpl<C> parser_;
        };

        // language
        //
        template <typename C>
        struct language_pimpl: virtual language_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual std::basic_string<C>
          post_language ();

        protected:
          std::basic_string<C> str_;
        };

        // anyURI
        //
        template <typename C>
        struct uri_pimpl: virtual uri_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual std::basic_string<C>
          post_uri ();

        protected:
          std::basic_string<C> str_;
        };

        // QName
        //
        template <typename C>
        struct qname_pimpl: virtual qname_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual qname<C>
          post_qname ();

        protected:
          std::basic_string<C> str_;
          std::basic_string<C> name_;
          std::basic_string<C> prefix_;
        };

        // base64Binary
        //
        template <typename C>
        struct base64_binary_pimpl: virtual base64_binary_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual std::auto_ptr<buffer>
          post_base64_binary ();

        protected:
          std::basic_string<C> str_;
          std::auto_ptr<buffer> buf_;
        };

        // hexBinary
        //
        template <typename C>
        struct hex_binary_pimpl: virtual hex_binary_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual std::auto_ptr<buffer>
          post_hex_binary ();

        protected:
          std::basic_string<C> str_;
          std::auto_ptr<buffer> buf_;
        };

        // gday
        //
        template <typename C>
        struct gday_pimpl: virtual gday_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual gday
          post_gday ();

        protected:
          std::basic_string<C> str_;
          unsigned short day_;
          bool z_;
          short zh_, zm_;
        };

        // gmonth
        //
        template <typename C>
        struct gmonth_pimpl: virtual gmonth_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual gmonth
          post_gmonth ();

        protected:
          std::basic_string<C> str_;
          unsigned short month_;
          bool z_;
          short zh_, zm_;
        };

        // gyear
        //
        template <typename C>
        struct gyear_pimpl: virtual gyear_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual gyear
          post_gyear ();

        protected:
          std::basic_string<C> str_;
          int year_;
          bool z_;
          short zh_, zm_;
        };

        // gmonth_day
        //
        template <typename C>
        struct gmonth_day_pimpl: virtual gmonth_day_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual gmonth_day
          post_gmonth_day ();

        protected:
          std::basic_string<C> str_;
          unsigned short month_;
          unsigned short day_;
          bool z_;
          short zh_, zm_;
        };

        // gyear_month
        //
        template <typename C>
        struct gyear_month_pimpl: virtual gyear_month_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual gyear_month
          post_gyear_month ();

        protected:
          std::basic_string<C> str_;
          int year_;
          unsigned short month_;
          bool z_;
          short zh_, zm_;
        };

        // date
        //
        template <typename C>
        struct date_pimpl: virtual date_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual date
          post_date ();

        protected:
          std::basic_string<C> str_;
          int year_;
          unsigned short month_;
          unsigned short day_;
          bool z_;
          short zh_, zm_;
        };

        // time
        //
        template <typename C>
        struct time_pimpl: virtual time_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual time
          post_time ();

        protected:
          std::basic_string<C> str_;
          unsigned short hours_;
          unsigned short minutes_;
          double seconds_;
          bool z_;
          short zh_, zm_;
        };

        // date_time
        //
        template <typename C>
        struct date_time_pimpl: virtual date_time_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual date_time
          post_date_time ();

        protected:
          std::basic_string<C> str_;
          int year_;
          unsigned short month_;
          unsigned short day_;
          unsigned short hours_;
          unsigned short minutes_;
          double seconds_;
          bool z_;
          short zh_, zm_;
        };

        // duration
        //
        template <typename C>
        struct duration_pimpl: virtual duration_pskel<C>
        {
          virtual void
          _pre ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post ();

          virtual duration
          post_duration ();

        protected:
          std::basic_string<C> str_;
          bool negative_;
          unsigned int years_;
          unsigned int months_;
          unsigned int days_;
          unsigned int hours_;
          unsigned int minutes_;
          double seconds_;
        };

        // Literals.
        //
        namespace bits
        {
          template<typename C>
          const C*
          boolean ();

          template<typename C>
          const C*
          byte ();

          template<typename C>
          const C*
          unsigned_byte ();

          template<typename C>
          const C*
          short_ ();

          template<typename C>
          const C*
          unsigned_short ();

          template<typename C>
          const C*
          int_ ();

          template<typename C>
          const C*
          unsigned_int ();

          template<typename C>
          const C*
          long_ ();

          template<typename C>
          const C*
          unsigned_long ();

          template<typename C>
          const C*
          integer ();

          template<typename C>
          const C*
          negative_integer ();

          template<typename C>
          const C*
          non_positive_integer ();

          template<typename C>
          const C*
          non_negative_integer ();

          template<typename C>
          const C*
          positive_integer ();

          template<typename C>
          const C*
          float_ ();

          template<typename C>
          const C*
          double_ ();

          template<typename C>
          const C*
          decimal ();

          template<typename C>
          const C*
          name ();

          template<typename C>
          const C*
          nmtoken ();

          template<typename C>
          const C*
          nmtokens ();

          template<typename C>
          const C*
          ncname ();

          template<typename C>
          const C*
          id ();

          template<typename C>
          const C*
          idref ();

          template<typename C>
          const C*
          idrefs ();

          template<typename C>
          const C*
          language ();

          template<typename C>
          const C*
          qname ();

          template<typename C>
          const C*
          base64_binary ();

          template<typename C>
          const C*
          hex_binary ();

          template<typename C>
          const C*
          gday ();

          template<typename C>
          const C*
          gmonth ();

          template<typename C>
          const C*
          gyear ();

          template<typename C>
          const C*
          gmonth_day ();

          template<typename C>
          const C*
          gyear_month ();

          template<typename C>
          const C*
          date ();

          template<typename C>
          const C*
          time ();

          template<typename C>
          const C*
          date_time ();

          template<typename C>
          const C*
          duration ();

          // float literals: INF -INF NaN
          //
          template<typename C>
          const C*
          positive_inf ();

          template<typename C>
          const C*
          negative_inf ();

          template<typename C>
          const C*
          nan ();

          // boolean literals
          //
          template<typename C>
          const C*
          true_ ();

          template<typename C>
          const C*
          false_ ();

          template<typename C>
          const C*
          one ();

          template<typename C>
          const C*
          zero ();
        }
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/validating/xml-schema-pimpl.txx>

#endif  // XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PIMPL_HXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/validating/xml-schema-pimpl.ixx>
