// file      : xsd/cxx/tree/type-serializer-map.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_TYPE_SERIALIZER_MAP_HXX
#define XSD_CXX_TREE_TYPE_SERIALIZER_MAP_HXX

#include <map>
#include <string>
#include <cstddef>  // std::size_t
#include <typeinfo>

#include <xercesc/dom/DOMElement.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/qualified-name.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/serialization-header.hxx> // namespace_infomap

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      template <typename C>
      struct type_serializer_map
      {
        typedef std::type_info type_id;
        typedef xml::qualified_name<C> qualified_name;
        typedef void (*serializer) (xercesc::DOMElement&, const type&);

        void
        register_type (const type_id&,
                       const qualified_name& name,
                       serializer,
                       bool override = true);

        void
        unregister_type (const type_id&);

        void
        register_element (const qualified_name& root,
                          const qualified_name& subst,
                          const type_id&,
                          serializer);

        void
        unregister_element (const qualified_name& root, const type_id&);

      public:
        void
        serialize (const C* name, // element name
                   const C* ns,   // element namespace
                   bool global,
                   bool qualified,
                   xercesc::DOMElement& parent,
                   const type&) const;

        // Serialize into existing element.
        //
        void
        serialize (const C* static_name,
                   const C* static_ns,
                   xercesc::DOMElement&,
                   const qualified_name&,
                   const type&) const;

        // Create DOMDocument with root element suitable for serializing
        // x into it.
        //
        xml::dom::auto_ptr<xercesc::DOMDocument>
        serialize (const C* name, // element name
                   const C* ns,   // element namespace
                   const xml::dom::namespace_infomap<C>&,
                   const type& x,
                   unsigned long flags) const;

      public:
        type_serializer_map ();

      public:
        struct type_info
        {
          type_info (const qualified_name& name,
                     typename type_serializer_map::serializer serializer)
              : name_ (name), serializer_ (serializer)
          {
          }

          const qualified_name&
          name () const
          {
            return name_;
          }

          typename type_serializer_map::serializer
          serializer () const
          {
            return serializer_;
          }

          // For std::map.
          //
          type_info ()
              : name_ (std::basic_string<C> (), std::basic_string<C> ()),
                serializer_ (0)
          {
          }

        private:
          qualified_name name_;
          typename type_serializer_map::serializer serializer_;
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

        // Map of (root-element to map of (type_id to type_info)).
        // Note that here type_info::name is element name.
        //
        typedef
        std::map<const type_id*, type_info, type_id_comparator>
        subst_map;

        typedef
        std::map<qualified_name, subst_map>
        element_map;

        type_map type_map_;
        element_map element_map_;

      private:
        const type_info*
        find_substitution (const subst_map& start, const type_id&) const;

        // Sets an xsi:type attribute corresponding to the type_info.
        //
        void
        set_xsi_type (xercesc::DOMElement& parent,
                      xercesc::DOMElement&,
                      const type_info&) const;
      };


      //
      //
      template<unsigned long id, typename C>
      struct type_serializer_plate
      {
        static type_serializer_map<C>* map;
        static std::size_t count;

        type_serializer_plate ();
        ~type_serializer_plate ();
      };

      template<unsigned long id, typename C>
      type_serializer_map<C>* type_serializer_plate<id, C>::map = 0;

      template<unsigned long id, typename C>
      std::size_t type_serializer_plate<id, C>::count = 0;


      //
      //
      template<unsigned long id, typename C>
      inline type_serializer_map<C>&
      type_serializer_map_instance ()
      {
        return *type_serializer_plate<id, C>::map;
      }

      //
      //
      template<typename T>
      void
      serializer_impl (xercesc::DOMElement&, const type&);


      //
      //
      template<unsigned long id, typename C, typename T>
      struct type_serializer_initializer
      {
        type_serializer_initializer (const C* name, const C* ns);
        ~type_serializer_initializer ();
      };


      //
      //
      template<unsigned long id, typename C, typename T>
      struct element_serializer_initializer
      {
        element_serializer_initializer (const C* root_name, const C* root_ns,
                                        const C* subst_name, const C* subst_ns);

        ~element_serializer_initializer ();

      private:
        const C* root_name_;
        const C* root_ns_;
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/type-serializer-map.txx>

#endif // XSD_CXX_TREE_TYPE_SERIALIZER_MAP_HXX
