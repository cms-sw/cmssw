// file      : xsd/cxx/parser/validating/inheritance-map.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_VALIDATING_INHERITANCE_MAP_HXX
#define XSD_CXX_PARSER_VALIDATING_INHERITANCE_MAP_HXX

#include <map>
#include <cstddef> // std::size_t

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {
        template <typename C>
        struct string_comparison
        {
          bool
          operator() (const C* x, const C* y) const
          {
            ro_string<C> s (x);
            return s.compare (y) < 0;
          }
        };

        template <typename C>
        struct inheritance_map
        {
          void
          insert (const C* derived, const C* base)
          {
            map_[derived] = base;
          }

          void
          erase (const C* derived)
          {
            map_.erase (derived);
          }

          bool
          check (const C* derived, const ro_string<C>& base) const;

        private:
          typedef std::map<const C*, const C*, string_comparison<C> > map;
          map map_;
        };


        // Translation unit initializer.
        //
        template<typename C>
        struct inheritance_map_init
        {
          static inheritance_map<C>* map;
          static std::size_t count;

          inheritance_map_init ();
          ~inheritance_map_init ();
        };

        template<typename C>
        inheritance_map<C>* inheritance_map_init<C>::map = 0;

        template<typename C>
        std::size_t inheritance_map_init<C>::count = 0;

        template<typename C>
        inline inheritance_map<C>&
        inheritance_map_instance ()
        {
          return *inheritance_map_init<C>::map;
        }


        // Map entry initializer.
        //
        template<typename C>
        struct inheritance_map_entry
        {
          inheritance_map_entry (const C* derived, const C* base);
          ~inheritance_map_entry ();

        private:
          const C* derived_;
        };
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/validating/inheritance-map.txx>

#endif  // XSD_CXX_PARSER_VALIDATING_INHERITANCE_MAP_HXX
