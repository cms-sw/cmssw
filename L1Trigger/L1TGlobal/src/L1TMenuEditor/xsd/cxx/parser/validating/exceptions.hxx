// file      : xsd/cxx/parser/validating/exceptions.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_VALIDATING_EXCEPTIONS_HXX
#define XSD_CXX_PARSER_VALIDATING_EXCEPTIONS_HXX

#include <string>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/schema-exceptions.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {
        //
        //
        template <typename C>
        struct expected_attribute: schema_exception<C>
        {
          virtual
          ~expected_attribute ();

          expected_attribute (const std::basic_string<C>& expected_namespace,
                              const std::basic_string<C>& expected_name);

          const std::basic_string<C>&
          expected_namespace () const
          {
            return expected_namespace_;
          }

          const std::basic_string<C>&
          expected_name () const
          {
            return expected_name_;
          }

          virtual std::basic_string<C>
          message () const;

        private:
          std::basic_string<C> expected_namespace_;
          std::basic_string<C> expected_name_;
        };

        //
        //
        template <typename C>
        struct unexpected_attribute: schema_exception<C>
        {
          virtual
          ~unexpected_attribute ();

          unexpected_attribute (
            const std::basic_string<C>& encountered_namespace,
            const std::basic_string<C>& encountered_name);


          const std::basic_string<C>&
          encountered_namespace () const
          {
            return encountered_namespace_;
          }

          const std::basic_string<C>&
          encountered_name () const
          {
            return encountered_name_;
          }

          virtual std::basic_string<C>
          message () const;

        private:
          std::basic_string<C> encountered_namespace_;
          std::basic_string<C> encountered_name_;
        };


        //
        //
        template <typename C>
        struct unexpected_characters: schema_exception<C>
        {
          virtual
          ~unexpected_characters ();

          unexpected_characters (const std::basic_string<C>& s);

          const std::basic_string<C>&
          characters () const
          {
            return characters_;
          }

          virtual std::basic_string<C>
          message () const;

        private:
          std::basic_string<C> characters_;
        };

        //
        //
        template <typename C>
        struct invalid_value: schema_exception<C>
        {
          virtual
          ~invalid_value ();

          invalid_value (const C* type, const std::basic_string<C>& value);

          invalid_value (const C* type, const ro_string<C>& value);

          invalid_value (const std::basic_string<C>& type,
                         const std::basic_string<C>& value);

          const std::basic_string<C>&
          type () const
          {
            return type_;
          }

          const std::basic_string<C>&
          value () const
          {
            return value_;
          }

          virtual std::basic_string<C>
          message () const;

        private:
          std::basic_string<C> type_;
          std::basic_string<C> value_;
        };
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/validating/exceptions.txx>

#endif  // XSD_CXX_PARSER_VALIDATING_EXCEPTIONS_HXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/validating/exceptions.ixx>
