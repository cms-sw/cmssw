// file      : xsd/cxx/parser/map.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_MAP_HXX
#define XSD_CXX_PARSER_MAP_HXX

#include <map>
#include <string>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/elements.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // Parser map. Used in the polymorphic document parsing.
      //
      template <typename C>
      struct parser_map
      {
        virtual
        ~parser_map ();

        // The type argument is the type name and namespace from the
        // xsi:type attribute or substitution group map in the form
        // "<name> <namespace>" with the space and namespace part
        // absent if the type does not have a namespace.
        //
        virtual parser_base<C>*
        find (const ro_string<C>& type) const = 0;
      };


      // Parser map implementation.
      //
      template <typename C>
      struct parser_map_impl: parser_map<C>
      {
        parser_map_impl ();

        void
        insert (parser_base<C>&);

        virtual parser_base<C>*
        find (const ro_string<C>& type) const;

      private:
        parser_map_impl (const parser_map_impl&);

        parser_map_impl&
        operator= (const parser_map_impl&);

      private:
        struct string_comparison
        {
          bool
          operator() (const C* x, const C* y) const
          {
            ro_string<C> s (x);
            return s.compare (y) < 0;
          }
        };

        typedef std::map<const C*, parser_base<C>*, string_comparison> map;
        map map_;
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/map.ixx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/map.txx>

#endif  // XSD_CXX_PARSER_MAP_HXX
