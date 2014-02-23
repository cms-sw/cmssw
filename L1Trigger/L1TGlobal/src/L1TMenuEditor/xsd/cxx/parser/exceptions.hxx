// file      : xsd/cxx/parser/exceptions.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_EXCEPTIONS_HXX
#define XSD_CXX_PARSER_EXCEPTIONS_HXX

#include <string>
#include <vector>
#include <ostream>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/exceptions.hxx>       // xsd::cxx::exception
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      //
      //
      template <typename C>
      struct exception: xsd::cxx::exception
      {
        friend
        std::basic_ostream<C>&
        operator<< (std::basic_ostream<C>& os, const exception& e)
        {
          e.print (os);
          return os;
        }

      protected:
        virtual void
        print (std::basic_ostream<C>&) const = 0;
      };


      //
      //
      struct severity
      {
        enum value
        {
          warning,
          error
        };

        severity (value v) : v_ (v) {}
        operator value () const { return v_; }

      private:
        value v_;
      };

      template <typename C>
      struct error
      {
        error (cxx::parser::severity,
               const std::basic_string<C>& id,
               unsigned long line,
               unsigned long column,
               const std::basic_string<C>& message);

        cxx::parser::severity
        severity () const
        {
          return severity_;
        }

        const std::basic_string<C>&
        id () const
        {
          return id_;
        }

        unsigned long
        line () const
        {
          return line_;
        }

        unsigned long
        column () const
        {
          return column_;
        }

        const std::basic_string<C>&
        message () const
        {
          return message_;
        }

      private:
        cxx::parser::severity severity_;
        std::basic_string<C> id_;
        unsigned long line_;
        unsigned long column_;
        std::basic_string<C> message_;
      };

      // See exceptions.ixx for operator<< (error).


      //
      //
      template <typename C>
      struct diagnostics: std::vector<error<C> >
      {
      };

      // See exceptions.ixx for operator<< (diagnostics).

      //
      //
      template <typename C>
      struct parsing: exception<C>
      {
        virtual
        ~parsing () throw ();

        parsing ();

        parsing (const cxx::parser::diagnostics<C>&);

        const cxx::parser::diagnostics<C>&
        diagnostics () const
        {
          return diagnostics_;
        }

        virtual const char*
        what () const throw ();

      protected:
        virtual void
        print (std::basic_ostream<C>&) const;

      private:
        cxx::parser::diagnostics<C> diagnostics_;
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/exceptions.txx>

#endif  // XSD_CXX_PARSER_EXCEPTIONS_HXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/exceptions.ixx>
