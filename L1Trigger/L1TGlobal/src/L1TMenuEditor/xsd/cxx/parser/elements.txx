// file      : xsd/cxx/parser/elements.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // parser_base
      //
      template <typename C>
      parser_base<C>::
      ~parser_base ()
      {
      }

      template <typename C>
      void parser_base<C>::
      pre ()
      {
      }

      template <typename C>
      void parser_base<C>::
      _pre ()
      {
      }

      template <typename C>
      void parser_base<C>::
      _post ()
      {
      }

      template <typename C>
      void parser_base<C>::
      _pre_impl ()
      {
        _pre ();
      }

      template <typename C>
      void parser_base<C>::
      _post_impl ()
      {
        _post ();
      }

      template <typename C>
      const C* parser_base<C>::
      _dynamic_type () const
      {
        return 0;
      }
    }
  }
}
