// file      : xsd/cxx/parser/validating/inheritance-map.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {
        template <typename C>
        bool inheritance_map<C>::
        check (const C* derived, const ro_string<C>& base) const
        {
          if (base == derived)
            return true;

          typename map::const_iterator i (map_.find (derived));

          if (i != map_.end ())
          {
            if (base == i->second)
              return true;
            else
              return check (i->second, base);
          }

          return false;
        }

        // inheritance_map_init
        //
        template<typename C>
        inheritance_map_init<C>::
        inheritance_map_init ()
        {
          if (count == 0)
            map = new inheritance_map<C>;

          ++count;
        }

        template<typename C>
        inheritance_map_init<C>::
        ~inheritance_map_init ()
        {
          if (--count == 0)
            delete map;
        }

        // inheritance_map_entry
        //
        template<typename C>
        inheritance_map_entry<C>::
        inheritance_map_entry (const C* derived, const C* base)
            : derived_ (derived)
        {
          inheritance_map_instance<C> ().insert (derived, base);
        }

        template<typename C>
        inheritance_map_entry<C>::
        ~inheritance_map_entry ()
        {
          inheritance_map_instance<C> ().erase (derived_);
        }
      }
    }
  }
}
