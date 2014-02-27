// file      : xsd/cxx/tree/type-factory-map.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_TYPE_FACTORY_MAP_HXX
#define XSD_CXX_TREE_TYPE_FACTORY_MAP_HXX

#include <map>
#include <string>
#include <memory>  // std::auto_ptr
#include <cstddef> // std::size_t

#include <xercesc/dom/DOMElement.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/qualified-name.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      template <typename C>
      struct type_factory_map
      {
        typedef xml::qualified_name<C> qualified_name;
        typedef std::auto_ptr<type> (*factory) (const xercesc::DOMElement&,
                                                flags,
                                                container*);
      public:
        type_factory_map ();

        void
        register_type (const qualified_name& name,
                       factory,
                       bool override = true);

        void
        unregister_type (const qualified_name& name);

        void
        register_element (const qualified_name& root,
                          const qualified_name& subst,
                          factory);

        void
        unregister_element (const qualified_name& root,
                            const qualified_name& subst);

        std::auto_ptr<type>
        create (const C* name, // element name
                const C* ns,   // element namespace
                factory static_type,
                bool global,
                bool qualified,
                const xercesc::DOMElement&,
                const qualified_name&,
                flags,
                container*) const;

      public:
        factory
        find (const qualified_name& name) const;

      private:
        template <typename T>
        static std::auto_ptr<type>
        traits_adapter (const xercesc::DOMElement&, flags, container*);

      private:
        typedef
        std::map<qualified_name, factory>
        type_map;

        // Map of (root-element to map of (subst-element to factory)).
        //
        typedef
        std::map<qualified_name, factory>
        subst_map;

        typedef
        std::map<qualified_name, subst_map>
        element_map;

        type_map type_map_;
        element_map element_map_;

      private:
        factory
        find_substitution (const subst_map& start,
                           const qualified_name& name) const;

        // The name argument is as specified in xsi:type.
        //
        factory
        find_type (const std::basic_string<C>& name,
                   const xercesc::DOMElement&) const;
      };


      //
      //
      template<unsigned long id, typename C>
      struct type_factory_plate
      {
        static type_factory_map<C>* map;
        static std::size_t count;

        type_factory_plate ();
        ~type_factory_plate ();
      };

      template<unsigned long id, typename C>
      type_factory_map<C>* type_factory_plate<id, C>::map = 0;

      template<unsigned long id, typename C>
      std::size_t type_factory_plate<id, C>::count = 0;


      //
      //
      template<unsigned long id, typename C>
      inline type_factory_map<C>&
      type_factory_map_instance ()
      {
        return *type_factory_plate<id, C>::map;
      }


      //
      //
      template<typename T>
      std::auto_ptr<type>
      factory_impl (const xercesc::DOMElement&, flags, container*);

      //
      //
      template<unsigned long id, typename C, typename T>
      struct type_factory_initializer
      {
        type_factory_initializer (const C* name, const C* ns);
        ~type_factory_initializer ();

      private:
        const C* name_;
        const C* ns_;
      };

      //
      //
      template<unsigned long id, typename C, typename T>
      struct element_factory_initializer
      {
        element_factory_initializer (const C* root_name, const C* root_ns,
                                     const C* subst_name, const C* subst_ns);

        ~element_factory_initializer ();

      private:
        const C* root_name_;
        const C* root_ns_;
        const C* subst_name_;
        const C* subst_ns_;
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/type-factory-map.txx>

#endif // XSD_CXX_TREE_TYPE_FACTORY_MAP_HXX
