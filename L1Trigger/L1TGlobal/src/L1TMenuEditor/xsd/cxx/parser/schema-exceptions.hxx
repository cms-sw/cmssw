// file      : xsd/cxx/parser/schema-exceptions.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_SCHEMA_EXCEPTIONS_HXX
#define XSD_CXX_PARSER_SCHEMA_EXCEPTIONS_HXX

#include <string>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      template <typename C>
      struct schema_exception
      {
      public:
        virtual
        ~schema_exception ()
        {
        }

        unsigned long
        line () const
        {
          return line_;
        }

        void
        line (unsigned long l)
        {
          line_ = l;
        }

        unsigned long
        column () const
        {
          return column_;
        }

        void
        column (unsigned long c)
        {
          column_ = c;
        }

        const std::basic_string<C>&
        id () const
        {
          return id_;
        }

        void
        id (const std::basic_string<C>& id)
        {
          id_ = id;
        }

        virtual std::basic_string<C>
        message () const = 0;

      protected:
        unsigned long line_;
        unsigned long column_;
        std::basic_string<C> id_;
      };

      //
      //
      template <typename C>
      struct expected_element: schema_exception<C>
      {
        virtual
        ~expected_element ();

        expected_element (const std::basic_string<C>& expected_namespace,
                          const std::basic_string<C>& expected_name);

        expected_element (const std::basic_string<C>& expected_namespace,
                          const std::basic_string<C>& expected_name,
                          const std::basic_string<C>& encountered_namespace,
                          const std::basic_string<C>& encountered_name);

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

        // Encountered element namespace and name are empty if none
        // encountered.
        //
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
        std::basic_string<C> expected_namespace_;
        std::basic_string<C> expected_name_;

        std::basic_string<C> encountered_namespace_;
        std::basic_string<C> encountered_name_;
      };


      //
      //
      template <typename C>
      struct unexpected_element: schema_exception<C>
      {
        virtual
        ~unexpected_element ();

        unexpected_element (const std::basic_string<C>& encountered_namespace,
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
      struct dynamic_type: schema_exception<C>
      {
        virtual
        ~dynamic_type () throw ();

        dynamic_type (const std::basic_string<C>& type);

        const std::basic_string<C>&
        type () const
        {
          return type_;
        }

        virtual std::basic_string<C>
        message () const;

      private:
        std::basic_string<C> type_;
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/schema-exceptions.txx>

#endif  // XSD_CXX_PARSER_SCHEMA_EXCEPTIONS_HXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/schema-exceptions.ixx>
