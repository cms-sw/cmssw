// file      : xsd/cxx/tree/facet.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_FACET_HXX
#define XSD_CXX_TREE_FACET_HXX

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // We need to keep this type POD in order to be able to create a
      // static array.
      //
      struct facet
      {
        enum id_type
        {
          none,
          total_digits,
          fraction_digits
        };

        id_type id;
        unsigned long value;

        static const facet*
        find (const facet* facets, facet::id_type id)
        {
          while (facets->id != id && facets->id != none)
            ++facets;

          return facets->id != none ? facets : 0;
        }
      };
    }
  }
}

#endif  // XSD_CXX_TREE_FACET_HXX
