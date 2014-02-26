// file      : xsd/cxx/parser/validating/xml-schema-pskel.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PSKEL_HXX
#define XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PSKEL_HXX

#include <string>
#include <memory> // auto_ptr

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/xml-schema.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/validating/parser.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {
        // anyType and anySimpleType. All events are routed to the
        // _any_* callbacks.
        //
        template <typename C>
        struct any_type_pskel: complex_content<C>
        {
          virtual bool
          _start_element_impl (const ro_string<C>&,
                               const ro_string<C>&,
                               const ro_string<C>*);

          virtual bool
          _end_element_impl (const ro_string<C>&,
                             const ro_string<C>&);

          virtual bool
          _attribute_impl_phase_two (const ro_string<C>&,
                                     const ro_string<C>&,
                                     const ro_string<C>&);

          virtual bool
          _characters_impl (const ro_string<C>&);

          virtual void
          post_any_type () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct any_simple_type_pskel: simple_content<C>
        {
          virtual bool
          _characters_impl (const ro_string<C>&);

          virtual void
          post_any_simple_type () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };


        // Boolean.
        //
        template <typename C>
        struct boolean_pskel: simple_content<C>
        {
          virtual bool
          post_boolean () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };


        // 8-bit
        //
        template <typename C>
        struct byte_pskel: simple_content<C>
        {
          virtual signed char
          post_byte () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct unsigned_byte_pskel: simple_content<C>
        {
          virtual unsigned char
          post_unsigned_byte () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };


        // 16-bit
        //
        template <typename C>
        struct short_pskel: simple_content<C>
        {
          virtual short
          post_short () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct unsigned_short_pskel: simple_content<C>
        {
          virtual unsigned short
          post_unsigned_short () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };


        // 32-bit
        //
        template <typename C>
        struct int_pskel: simple_content<C>
        {
          virtual int
          post_int () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct unsigned_int_pskel: simple_content<C>
        {
          virtual unsigned int
          post_unsigned_int () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };


        // 64-bit
        //
        template <typename C>
        struct long_pskel: simple_content<C>
        {
          virtual long long
          post_long () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct unsigned_long_pskel: simple_content<C>
        {
          virtual unsigned long long
          post_unsigned_long () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };


        // Arbitrary-length integers.
        //
        template <typename C>
        struct integer_pskel: simple_content<C>
        {
          virtual long long
          post_integer () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct negative_integer_pskel: simple_content<C>
        {
          virtual long long
          post_negative_integer () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct non_positive_integer_pskel: simple_content<C>
        {
          virtual long long
          post_non_positive_integer () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct positive_integer_pskel: simple_content<C>
        {
          virtual unsigned long long
          post_positive_integer () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct non_negative_integer_pskel: simple_content<C>
        {
          virtual unsigned long long
          post_non_negative_integer () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };


        // Floats.
        //
        template <typename C>
        struct float_pskel: simple_content<C>
        {
          virtual float
          post_float () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct double_pskel: simple_content<C>
        {
          virtual double
          post_double () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct decimal_pskel: simple_content<C>
        {
          virtual double
          post_decimal () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };


        // Strings.
        //
        template <typename C>
        struct string_pskel: simple_content<C>
        {
          virtual std::basic_string<C>
          post_string () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct normalized_string_pskel: simple_content<C>
        {
          virtual std::basic_string<C>
          post_normalized_string () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct token_pskel: simple_content<C>
        {
          virtual std::basic_string<C>
          post_token () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct name_pskel: simple_content<C>
        {
          virtual std::basic_string<C>
          post_name () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct nmtoken_pskel: simple_content<C>
        {
          virtual std::basic_string<C>
          post_nmtoken () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct nmtokens_pskel: list_base<C>
        {
          virtual string_sequence<C>
          post_nmtokens () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct ncname_pskel: simple_content<C>
        {
          virtual std::basic_string<C>
          post_ncname () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct id_pskel: simple_content<C>
        {
          virtual std::basic_string<C>
          post_id () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct idref_pskel: simple_content<C>
        {
          virtual std::basic_string<C>
          post_idref () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct idrefs_pskel: list_base<C>
        {
          virtual string_sequence<C>
          post_idrefs () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        // Language.
        //
        template <typename C>
        struct language_pskel: simple_content<C>
        {
          virtual std::basic_string<C>
          post_language () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        // URI.
        //
        template <typename C>
        struct uri_pskel: simple_content<C>
        {
          virtual std::basic_string<C>
          post_uri () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        // QName.
        //
        template <typename C>
        struct qname_pskel: simple_content<C>
        {
          virtual qname<C>
          post_qname () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        // Base64 and hex binaries.
        //
        template <typename C>
        struct base64_binary_pskel: simple_content<C>
        {
          virtual std::auto_ptr<buffer>
          post_base64_binary () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct hex_binary_pskel: simple_content<C>
        {
          virtual std::auto_ptr<buffer>
          post_hex_binary () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        // Time and date types.
        //
        template <typename C>
        struct gday_pskel: simple_content<C>
        {
          virtual gday
          post_gday () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct gmonth_pskel: simple_content<C>
        {
          virtual gmonth
          post_gmonth () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct gyear_pskel: simple_content<C>
        {
          virtual gyear
          post_gyear () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct gmonth_day_pskel: simple_content<C>
        {
          virtual gmonth_day
          post_gmonth_day () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct gyear_month_pskel: simple_content<C>
        {
          virtual gyear_month
          post_gyear_month () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct date_pskel: simple_content<C>
        {
          virtual date
          post_date () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct time_pskel: simple_content<C>
        {
          virtual time
          post_time () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct date_time_pskel: simple_content<C>
        {
          virtual date_time
          post_date_time () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };

        template <typename C>
        struct duration_pskel: simple_content<C>
        {
          virtual duration
          post_duration () = 0;

          static const C*
          _static_type ();

          virtual const C*
          _dynamic_type () const;
        };
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/validating/xml-schema-pskel.txx>

#endif  // XSD_CXX_PARSER_VALIDATING_XML_SCHEMA_PSKEL_HXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/validating/xml-schema-pskel.ixx>
