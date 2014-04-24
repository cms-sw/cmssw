// file      : xsd/cxx/tree/comparison-map.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_COMPARISON_MAP_HXX
#define XSD_CXX_TREE_COMPARISON_MAP_HXX

#include <map>
#include <cstddef>  // std::size_t
#include <typeinfo>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      template <typename C>
      struct comparison_map
      {
        typedef std::type_info type_id;
        typedef bool (*comparator) (const type&, const type&);

        comparison_map ();

        void
        register_type (const type_id&, comparator, bool override = true);

        void
        unregister_type (const type_id&);

        bool
        compare (const type&, const type&);

      public:
        comparator
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
        std::map<const type_id*, comparator, type_id_comparator>
        type_map;

        type_map type_map_;
      };

      //
      //
      template<unsigned long id, typename C>
      struct comparison_plate
      {
        static comparison_map<C>* map;
        static std::size_t count;

        comparison_plate ();
        ~comparison_plate ();
      };

      template<unsigned long id, typename C>
      comparison_map<C>* comparison_plate<id, C>::map = 0;

      template<unsigned long id, typename C>
      std::size_t comparison_plate<id, C>::count = 0;


      //
      //
      template<unsigned long id, typename C>
      inline comparison_map<C>&
      comparison_map_instance ()
      {
        return *comparison_plate<id, C>::map;
      }

      //
      //
      template<typename T>
      bool
      comparator_impl (const type&, const type&);

      template<unsigned long id, typename C, typename T>
      struct comparison_initializer
      {
        comparison_initializer ();
        ~comparison_initializer ();
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/comparison-map.txx>

#endif // XSD_CXX_TREE_COMPARISON_MAP_HXX
