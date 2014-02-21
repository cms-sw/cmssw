// file      : xsd/cxx/parser/error-handler.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      template <typename C>
      bool error_handler<C>::
      handle (const std::basic_string<C>& id,
              unsigned long line,
              unsigned long column,
              severity s,
              const std::basic_string<C>& message)
      {
        diagnostics_.push_back (
          error<C> (s == severity::warning
                    ? cxx::parser::severity::warning
                    : cxx::parser::severity::error,
                    id, line, column, message));

        if (!failed_ && s != severity::warning)
          failed_ = true;

        return true;
      }

      template <typename C>
      void error_handler<C>::
      throw_if_failed () const
      {
        if (failed_)
          throw parsing<C> (diagnostics_);
      }
    }
  }
}
