// file      : xsd/cxx/tree/stream-extraction-map.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_STREAM_EXTRACTION_MAP_HXX
#define XSD_CXX_TREE_STREAM_EXTRACTION_MAP_HXX

#include <map>
#include <memory>  // std::auto_ptr
#include <cstddef> // std::size_t

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/istream.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/qualified-name.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      template <typename S, typename C>
      struct stream_extraction_map
      {
        typedef xml::qualified_name<C> qualified_name;
        typedef std::auto_ptr<type> (*extractor) (
          istream<S>&, flags, container*);

      public:
        stream_extraction_map ();

        void
        register_type (const qualified_name& name,
                       extractor,
                       bool override = true);

        void
        unregister_type (const qualified_name& name);

        std::auto_ptr<type>
        extract (istream<S>&, flags, container*);

      public:
        extractor
        find (const qualified_name& name) const;

      private:
        typedef std::map<qualified_name, extractor> type_map;

        type_map type_map_;
      };

      //
      //
      template<unsigned long id, typename S, typename C>
      struct stream_extraction_plate
      {
        static stream_extraction_map<S, C>* map;
        static std::size_t count;

        stream_extraction_plate ();
        ~stream_extraction_plate ();
      };

      template<unsigned long id, typename S, typename C>
      stream_extraction_map<S, C>* stream_extraction_plate<id, S, C>::map = 0;

      template<unsigned long id, typename S, typename C>
      std::size_t stream_extraction_plate<id, S, C>::count = 0;


      //
      //
      template<unsigned long id, typename S, typename C>
      inline stream_extraction_map<S, C>&
      stream_extraction_map_instance ()
      {
        return *stream_extraction_plate<id, S, C>::map;
      }

      //
      //
      template<typename S, typename T>
      std::auto_ptr<type>
      extractor_impl (istream<S>&, flags, container*);


      template<unsigned long id, typename S, typename C, typename T>
      struct stream_extraction_initializer
      {
        stream_extraction_initializer (const C* name, const C* ns);
        ~stream_extraction_initializer ();

      private:
        const C* name_;
        const C* ns_;
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/stream-extraction-map.txx>

#endif // XSD_CXX_TREE_STREAM_EXTRACTION_MAP_HXX
