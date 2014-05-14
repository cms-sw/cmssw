// file      : xsd/cxx/tree/element-map.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // element_map
      //
      template <typename C, typename T>
      void element_map<C, T>::
      register_parser (const qualified_name& n, parser p)
      {
        (*map_)[n].parser_ = p;
      }

      template <typename C, typename T>
      void element_map<C, T>::
      register_serializer (const qualified_name& n, serializer s)
      {
        (*map_)[n].serializer_ = s;
      }

      // element_map_init
      //
      template <typename C, typename T>
      element_map_init<C, T>::
      element_map_init ()
      {
        if (element_map<C, T>::count_ == 0)
          element_map<C, T>::map_ = new typename element_map<C, T>::map;

        ++element_map<C, T>::count_;
      }

      template <typename C, typename T>
      element_map_init<C, T>::
      ~element_map_init ()
      {
        if (--element_map<C, T>::count_ == 0)
          delete element_map<C, T>::map_;
      }

      // parser_init
      //
      template<typename T, typename C, typename B>
      parser_init<T, C, B>::
      parser_init (const std::basic_string<C>& name,
                   const std::basic_string<C>& ns)
      {
        element_map<C, B>::register_parser (
          xml::qualified_name<C> (name, ns), &parser_impl<T, C, B>);
      }

      // serializer_init
      //
      template<typename T, typename C, typename B>
      serializer_init<T, C, B>::
      serializer_init (const std::basic_string<C>& name,
                   const std::basic_string<C>& ns)
      {
        element_map<C, B>::register_serializer (
          xml::qualified_name<C> (name, ns), &serializer_impl<T, C, B>);
      }
    }
  }
}
