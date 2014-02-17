// file      : xsd/cxx/parser/substitution-map.txx
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
      bool substitution_map<C>::
      check_ (const ro_string<C>& ns,
              const ro_string<C>& name,
              const C* root_ns,
              const C* root_name,
              const ro_string<C>** type) const
      {
        key k (ns, name);
        typename map::const_iterator i (map_.find (k));

        if (i == map_.end ())
          return false;

        const value& v (i->second);

        bool r (false);

        if (v.name () == root_name && v.ns () == root_ns)
          r = true;
        else
          r = check_ (v.ns (), v.name (), root_ns, root_name, 0);

        if (r && type != 0 && *type == 0)
          *type = &v.type ();

        return r;
      }

      // substitution_map_init
      //
      template<typename C>
      substitution_map_init<C>::
      substitution_map_init ()
      {
        if (count == 0)
          map = new substitution_map<C>;

        ++count;
      }

      template<typename C>
      substitution_map_init<C>::
      ~substitution_map_init ()
      {
        if (--count == 0)
          delete map;
      }

      // substitution_map_entry
      //
      template<typename C>
      substitution_map_entry<C>::
      substitution_map_entry (const C* member_ns,
                              const C* member_name,
                              const C* root_ns,
                              const C* root_name,
                              const C* member_type)
          : member_ns_ (member_ns), member_name_ (member_name)
      {
        substitution_map_instance<C> ().insert (
          member_ns, member_name, root_ns, root_name, member_type);
      }

      template<typename C>
      substitution_map_entry<C>::
      ~substitution_map_entry ()
      {
        substitution_map_instance<C> ().erase (member_ns_, member_name_);
      }
    }
  }
}
