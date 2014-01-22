// file      : xsd/cxx/tree/std-ostream-map.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_STD_OSTREAM_MAP_HXX
#define XSD_CXX_TREE_STD_OSTREAM_MAP_HXX

#include <map>
#include <cstddef>  // std::size_t
#include <ostream>
#include <typeinfo>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      template <typename C>
      struct std_ostream_map
      {
        typedef std::type_info type_id;
        typedef void (*inserter) (std::basic_ostream<C>&, const type&);

        std_ostream_map ();

        void
        register_type (const type_id&, inserter, bool override = true);

        void
        unregister_type (const type_id&);

        void
        insert (std::basic_ostream<C>&, const type&);

      public:
        inserter
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
        std::map<const type_id*, inserter, type_id_comparator>
        type_map;

        type_map type_map_;
      };

      //
      //
      template<unsigned long id, typename C>
      struct std_ostream_plate
      {
        static std_ostream_map<C>* map;
        static std::size_t count;

        std_ostream_plate ();
        ~std_ostream_plate ();
      };

      template<unsigned long id, typename C>
      std_ostream_map<C>* std_ostream_plate<id, C>::map = 0;

      template<unsigned long id, typename C>
      std::size_t std_ostream_plate<id, C>::count = 0;


      //
      //
      template<unsigned long id, typename C>
      inline std_ostream_map<C>&
      std_ostream_map_instance ()
      {
        return *std_ostream_plate<id, C>::map;
      }

      //
      //
      template<typename C, typename T>
      void
      inserter_impl (std::basic_ostream<C>&, const type&);

      template<unsigned long id, typename C, typename T>
      struct std_ostream_initializer
      {
        std_ostream_initializer ();
        ~std_ostream_initializer ();
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/std-ostream-map.txx>

#endif // XSD_CXX_TREE_STD_OSTREAM_MAP_HXX
