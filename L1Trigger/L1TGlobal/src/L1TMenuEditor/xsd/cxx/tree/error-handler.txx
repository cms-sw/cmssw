// file      : xsd/cxx/tree/error-handler.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace tree
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
                    ? tree::severity::warning
                    : tree::severity::error, id, line, column, message));

        if (!failed_ && s != severity::warning)
          failed_ = true;

        return true;
      }
    }
  }
}
