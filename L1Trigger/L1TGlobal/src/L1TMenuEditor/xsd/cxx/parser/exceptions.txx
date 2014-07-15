// file      : xsd/cxx/parser/exceptions.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // error
      //
      template <typename C>
      error<C>::
      error (cxx::parser::severity s,
             const std::basic_string<C>& id,
             unsigned long line,
             unsigned long column,
             const std::basic_string<C>& message)
          : severity_ (s),
            id_ (id),
            line_ (line),
            column_ (column),
            message_ (message)
      {
      }


      // parsing
      //
      template <typename C>
      parsing<C>::
      ~parsing () throw ()
      {
      }

      template <typename C>
      parsing<C>::
      parsing ()
      {
      }

      template <typename C>
      parsing<C>::
      parsing (const cxx::parser::diagnostics<C>& diagnostics)
          : diagnostics_ (diagnostics)
      {
      }

      template <typename C>
      const char* parsing<C>::
      what () const throw ()
      {
        return "instance document parsing failed";
      }
    }
  }
}
