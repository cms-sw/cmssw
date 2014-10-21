// file      : xsd/cxx/parser/substitution-map.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_SUBSTITUTION_MAP_HXX
#define XSD_CXX_PARSER_SUBSTITUTION_MAP_HXX

#include <map>
#include <cstddef> // std::size_t

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      template <typename C>
      struct substitution_map_key
      {
        substitution_map_key (const C* ns, const C* name)
            : ns_ (ns), name_ (name)
        {
        }

        substitution_map_key (const ro_string<C>& ns,
                              const ro_string<C>& name)
            : ns_ (ns.data (), ns.size ()),
              name_ (name.data (), name.size ())
        {
        }

        substitution_map_key (const substitution_map_key& x)
            : ns_ (x.ns_.data (), x.ns_.size ()),
              name_ (x.name_.data (), x.name_.size ())
        {
        }

      private:
        substitution_map_key&
        operator= (const substitution_map_key&);

      public:
        const ro_string<C>&
        ns () const
        {
          return ns_;
        }

        const ro_string<C>&
        name () const
        {
          return name_;
        }

      private:
        const ro_string<C> ns_;
        const ro_string<C> name_;
      };

      template <typename C>
      inline bool
      operator< (const substitution_map_key<C>& x,
                 const substitution_map_key<C>& y)
      {
        int r (x.name ().compare (y.name ()));
        return r < 0 || (r == 0 && x.ns () < y.ns ());
      }

      template <typename C>
      struct substitution_map_value
      {
        substitution_map_value (const C* ns, const C* name, const C* type)
            : ns_ (ns), name_ (name), type_ (type)
        {
        }

        substitution_map_value (const substitution_map_value& x)
            : ns_ (x.ns_.data (), x.ns_.size ()),
              name_ (x.name_.data (), x.name_.size ()),
              type_ (x.type_.data (), x.type_.size ())
        {
        }

        substitution_map_value&
        operator= (const substitution_map_value& x)
        {
          if (this != &x)
          {
            ns_.assign (x.ns_.data (), x.ns_.size ());
            name_.assign (x.name_.data (), x.name_.size ());
            type_.assign (x.type_.data (), x.type_.size ());
          }

          return *this;
        }

      public:
        const ro_string<C>&
        ns () const
        {
          return ns_;
        }

        const ro_string<C>&
        name () const
        {
          return name_;
        }

        const ro_string<C>&
        type () const
        {
          return type_;
        }

      private:
        ro_string<C> ns_;
        ro_string<C> name_;
        ro_string<C> type_;
      };

      template <typename C>
      struct substitution_map
      {
        void
        insert (const C* member_ns,
                const C* member_name,
                const C* root_ns,
                const C* root_name,
                const C* member_type)
        {
          key k (member_ns, member_name);
          value v (root_ns, root_name, member_type);
          map_.insert (std::pair<key, value> (k, v));
        }

        void
        erase (const C* member_ns, const C* member_name)
        {
          map_.erase (key (member_ns, member_name));
        }

        // Check and get the type set if found.
        //
        bool
        check (const ro_string<C>& ns,
               const ro_string<C>& name,
               const C* root_ns,
               const C* root_name,
               const ro_string<C>*& type) const
        {

          return map_.empty ()
            ? false
            : check_ (ns, name, root_ns, root_name, &type);
        }

        // Check but don't care about the type.
        //
        bool
        check (const ro_string<C>& ns,
               const ro_string<C>& name,
               const C* root_ns,
               const C* root_name) const
        {

          return map_.empty ()
            ? false
            : check_ (ns, name, root_ns, root_name, 0);
        }

      private:
        bool
        check_ (const ro_string<C>& ns,
                const ro_string<C>& name,
                const C* root_ns,
                const C* root_name,
                const ro_string<C>** type) const;

      private:
        typedef substitution_map_key<C> key;
        typedef substitution_map_value<C> value;
        typedef std::map<key, value> map;

        map map_;
      };


      // Translation unit initializer.
      //
      template<typename C>
      struct substitution_map_init
      {
        static substitution_map<C>* map;
        static std::size_t count;

        substitution_map_init ();
        ~substitution_map_init ();
      };

      template<typename C>
      substitution_map<C>* substitution_map_init<C>::map = 0;

      template<typename C>
      std::size_t substitution_map_init<C>::count = 0;

      template<typename C>
      inline substitution_map<C>&
      substitution_map_instance ()
      {
        return *substitution_map_init<C>::map;
      }


      // Map entry initializer.
      //
      template<typename C>
      struct substitution_map_entry
      {
        substitution_map_entry (const C* member_ns,
                                const C* member_name,
                                const C* root_ns,
                                const C* root_name,
                                const C* member_type);

        ~substitution_map_entry ();

      private:
        const C* member_ns_;
        const C* member_name_;
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/substitution-map.txx>

#endif  // XSD_CXX_PARSER_SUBSTITUTION_MAP_HXX
