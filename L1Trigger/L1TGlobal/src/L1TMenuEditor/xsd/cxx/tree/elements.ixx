// file      : xsd/cxx/tree/elements.ixx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // simple_type
      //

      template <typename B>
      inline simple_type<B>::
      simple_type ()
      {
      }

      template <typename B>
      template <typename C>
      inline simple_type<B>::
      simple_type (const C*)
      {
      }
    }
  }
}
