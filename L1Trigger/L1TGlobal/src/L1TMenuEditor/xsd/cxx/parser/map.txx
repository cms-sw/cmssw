// file      : xsd/cxx/parser/map.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // parser_map
      //
      template <typename C>
      parser_map<C>::
      ~parser_map ()
      {
      }

      // parser_map_impl
      //
      template <typename C>
      parser_base<C>* parser_map_impl<C>::
      find (const ro_string<C>& type) const
      {
        typename map::const_iterator i (map_.find (type.data ()));
        return i != map_.end () ? i->second : 0;
      }
    }
  }
}
