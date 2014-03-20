// file      : xsd/cxx/tree/element-map.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_ELEMENT_MAP_HXX
#define XSD_CXX_TREE_ELEMENT_MAP_HXX

#include <map>
#include <memory>   // std::auto_ptr
#include <cstddef>  // std::size_t
#include <string>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/qualified-name.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      /**
       * @brief Root element map.
       *
       * This class allows uniform parsing/serialization of multiple
       * root elements via the element_type base class.
       *
       * @nosubgrouping
       */
      template <typename C, typename T>
      class element_map
      {
      public:
        /**
         * @brief Common base class for all element types.
         */
        typedef tree::element_type<C, T> element_type;

        /**
         * @brief Parse a DOM element.
         *
         * @param e A DOM element to parse.
         * @param f Flags to create the new element object with.
         * @return An automatic pointer to the new element object.
         */
        static std::auto_ptr<element_type>
        parse (const xercesc::DOMElement& e, flags f = 0);

        /**
         * @brief Serialize an element object to a DOM element.
         *
         * @param e A DOM element to serialize to.
         * @param x An element object to serialize.
         */
        static void
        serialize (xercesc::DOMElement& e, const element_type& x);

      public:
        //@cond

        typedef xml::qualified_name<C> qualified_name;

        typedef std::auto_ptr<element_type>
        (*parser) (const xercesc::DOMElement&, flags f);

        typedef void
        (*serializer) (xercesc::DOMElement&, const element_type&);

        static void
        register_parser (const qualified_name&, parser);

        static void
        register_serializer (const qualified_name&, serializer);

      public:
        struct map_entry
        {
          map_entry () : parser_ (0), serializer_ (0) {}

          parser parser_;
          serializer serializer_;
        };

        typedef
        std::map<qualified_name, map_entry>
        map;

        static map* map_;
        static std::size_t count_;

      private:
        element_map ();

        //@endcond
      };

      //@cond

      template <typename C, typename T>
      typename element_map<C, T>::map* element_map<C, T>::map_ = 0;

      template <typename C, typename T>
      std::size_t element_map<C, T>::count_ = 0;

      template <typename C, typename T>
      struct element_map_init
      {
        element_map_init ();
        ~element_map_init ();
      };

      //
      //
      template<typename T, typename C, typename B>
      std::auto_ptr<element_type<C, B> >
      parser_impl (const xercesc::DOMElement&, flags);

      template<typename T, typename C, typename B>
      struct parser_init
      {
        parser_init (const std::basic_string<C>& name,
                     const std::basic_string<C>& ns);
      };

      //
      //
      template<typename T, typename C, typename B>
      void
      serializer_impl (xercesc::DOMElement&, const element_type<C, B>&);

      template<typename T, typename C, typename B>
      struct serializer_init
      {
        serializer_init (const std::basic_string<C>& name,
                         const std::basic_string<C>& ns);
      };

      //@endcond
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/element-map.txx>

#endif // XSD_CXX_TREE_ELEMENT_MAP_HXX
