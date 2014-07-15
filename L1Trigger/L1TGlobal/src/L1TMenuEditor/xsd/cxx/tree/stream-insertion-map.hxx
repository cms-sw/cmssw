// file      : xsd/cxx/tree/stream-insertion-map.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_STREAM_INSERTION_MAP_HXX
#define XSD_CXX_TREE_STREAM_INSERTION_MAP_HXX

#include <map>
#include <string>
#include <cstddef>  // std::size_t
#include <typeinfo>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/ostream.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/qualified-name.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      template <typename S, typename C>
      struct stream_insertion_map
      {
        typedef std::type_info type_id;
        typedef xml::qualified_name<C> qualified_name;
        typedef void (*inserter) (ostream<S>&, const type&);

        stream_insertion_map ();

        void
        register_type (const type_id&,
                       const qualified_name& name,
                       inserter,
                       bool override = true);

        void
        unregister_type (const type_id&);

        void
        insert (ostream<S>&, const type&);

      public:
        struct type_info
        {
          type_info (const qualified_name& name,
                     typename stream_insertion_map::inserter inserter)
              : name_ (name), inserter_ (inserter)
          {
          }

          const qualified_name&
          name () const
          {
            return name_;
          }

          typename stream_insertion_map::inserter
          inserter () const
          {
            return inserter_;
          }

          // For std::map.
          //
          type_info ()
              : name_ (std::basic_string<C> (), std::basic_string<C> ()),
                inserter_ (0)
          {
          }

        private:
          qualified_name name_;
          typename stream_insertion_map::inserter inserter_;
        };

      public:
        const type_info*
        find (const type_id&) const;

      private:
        struct type_id_comparator
        {
          bool
          operator() (const type_id* x, const type_id* y) const
          {
            // XL C++ on AIX has buggy type_info::before() in that
            // it returns true for two different type_info objects
            // that happened to be for the same type.
            //
#if defined(__xlC__) && defined(_AIX)
            return *x != *y && x->before (*y);
#else
            return x->before (*y);
#endif
          }
        };

        typedef
        std::map<const type_id*, type_info, type_id_comparator>
        type_map;

        type_map type_map_;
      };

      //
      //
      template<unsigned long id, typename S, typename C>
      struct stream_insertion_plate
      {
        static stream_insertion_map<S, C>* map;
        static std::size_t count;

        stream_insertion_plate ();
        ~stream_insertion_plate ();
      };

      template<unsigned long id, typename S, typename C>
      stream_insertion_map<S, C>* stream_insertion_plate<id, S, C>::map = 0;

      template<unsigned long id, typename S, typename C>
      std::size_t stream_insertion_plate<id, S, C>::count = 0;


      //
      //
      template<unsigned long id, typename S, typename C>
      inline stream_insertion_map<S, C>&
      stream_insertion_map_instance ()
      {
        return *stream_insertion_plate<id, S, C>::map;
      }

      //
      //
      template<typename S, typename T>
      void
      inserter_impl (ostream<S>&, const type&);

      template<unsigned long id, typename S, typename C, typename T>
      struct stream_insertion_initializer
      {
        stream_insertion_initializer (const C* name, const C* ns);
        ~stream_insertion_initializer ();
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/stream-insertion-map.txx>

#endif // XSD_CXX_TREE_STREAM_INSERTION_MAP_HXX
