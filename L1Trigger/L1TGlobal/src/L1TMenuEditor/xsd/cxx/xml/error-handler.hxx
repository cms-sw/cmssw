// file      : xsd/cxx/xml/error-handler.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_ERROR_HANDLER_HXX
#define XSD_CXX_XML_ERROR_HANDLER_HXX

#include <string>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      template <typename C>
      class error_handler
      {
      public:
        virtual
        ~error_handler ()
        {
        }

      public:

        // The fatal severity level results in termination
        // of the parsing process no matter what is returned
        // from handle.
        //
        struct severity
        {
          enum value
          {
            warning,
            error,
            fatal
          };

          severity (value v) : v_ (v) {}
          operator value () const { return v_; }

        private:
          value v_;
        };

        virtual bool
        handle (const std::basic_string<C>& id,
                unsigned long line,
                unsigned long column,
                severity,
                const std::basic_string<C>& message) = 0;
      };
    }
  }
}

#endif  // XSD_CXX_XML_ERROR_HANDLER_HXX
