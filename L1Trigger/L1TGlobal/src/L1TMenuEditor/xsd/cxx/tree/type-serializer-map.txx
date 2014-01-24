// file      : xsd/cxx/tree/type-serializer-map.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <xercesc/util/XMLUni.hpp>
#include <xercesc/validators/schema/SchemaSymbols.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/bits/literals.hxx> // xml::bits::{xsi_namespace, type}
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/serialization-source.hxx> // dom::{create_*, prefix}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/bits/literals.hxx>

#include <iostream>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // type_serializer_map
      //
      template <typename C>
      type_serializer_map<C>::
      type_serializer_map ()
      {
        // Register serializers for built-in non-fundamental types.
        //
        std::basic_string<C> xsd (bits::xml_schema<C> ());


        // anyType and anySimpleType.
        //
        register_type (
          typeid (type),
          qualified_name (bits::any_type<C> (), xsd),
          &serializer_impl<type>,
          false);

        typedef simple_type<type> simple_type;
        register_type (
          typeid (simple_type),
          qualified_name (bits::any_simple_type<C> (), xsd),
          &serializer_impl<simple_type>,
          false);


        // Strings
        //
        typedef string<C, simple_type> string;
        register_type (
          typeid (string),
          qualified_name (bits::string<C> (), xsd),
          &serializer_impl<string>,
          false);

        typedef normalized_string<C, string> normalized_string;
        register_type (
          typeid (normalized_string),
          qualified_name (bits::normalized_string<C> (), xsd),
          &serializer_impl<normalized_string>,
          false);

        typedef token<C, normalized_string> token;
        register_type (
          typeid (token),
          qualified_name (bits::token<C> (), xsd),
          &serializer_impl<token>,
          false);

        typedef name<C, token> name;
        register_type (
          typeid (name),
          qualified_name (bits::name<C> (), xsd),
          &serializer_impl<name>,
          false);

        typedef nmtoken<C, token> nmtoken;
        register_type (
          typeid (nmtoken),
          qualified_name (bits::nmtoken<C> (), xsd),
          &serializer_impl<nmtoken>,
          false);

        typedef nmtokens<C, simple_type, nmtoken> nmtokens;
        register_type (
          typeid (nmtokens),
          qualified_name (bits::nmtokens<C> (), xsd),
          &serializer_impl<nmtokens>,
          false);

        typedef ncname<C, name> ncname;
        register_type (
          typeid (ncname),
          qualified_name (bits::ncname<C> (), xsd),
          &serializer_impl<ncname>,
          false);

        typedef language<C, token> language;
        register_type (
          typeid (language),
          qualified_name (bits::language<C> (), xsd),
          &serializer_impl<language>,
          false);


        // ID/IDREF.
        //
        typedef id<C, ncname> id;
        register_type (
          typeid (id),
          qualified_name (bits::id<C> (), xsd),
          &serializer_impl<id>,
          false);

        typedef idref<C, ncname, type> idref;
        register_type (
          typeid (idref),
          qualified_name (bits::idref<C> (), xsd),
          &serializer_impl<idref>,
          false);

        typedef idrefs<C, simple_type, idref> idrefs;
        register_type (
          typeid (idrefs),
          qualified_name (bits::idrefs<C> (), xsd),
          &serializer_impl<idrefs>,
          false);


        // URI.
        //
        typedef uri<C, simple_type> uri;
        register_type (
          typeid (uri),
          qualified_name (bits::any_uri<C> (), xsd),
          &serializer_impl<uri>,
          false);


        // Qualified name.
        //
        typedef qname<C, simple_type, uri, ncname> qname;
        register_type (
          typeid (qname),
          qualified_name (bits::qname<C> (), xsd),
          &serializer_impl<qname>,
          false);


        // Binary.
        //
        typedef base64_binary<C, simple_type> base64_binary;
        register_type (
          typeid (base64_binary),
          qualified_name (bits::base64_binary<C> (), xsd),
          &serializer_impl<base64_binary>,
          false);

        typedef hex_binary<C, simple_type> hex_binary;
        register_type (
          typeid (hex_binary),
          qualified_name (bits::hex_binary<C> (), xsd),
          &serializer_impl<hex_binary>,
          false);


        // Date/time.
        //
        typedef gday<C, simple_type> gday;
        register_type (
          typeid (gday),
          qualified_name (bits::gday<C> (), xsd),
          &serializer_impl<gday>,
          false);

        typedef gmonth<C, simple_type> gmonth;
        register_type (
          typeid (gmonth),
          qualified_name (bits::gmonth<C> (), xsd),
          &serializer_impl<gmonth>,
          false);

        typedef gyear<C, simple_type> gyear;
        register_type (
          typeid (gyear),
          qualified_name (bits::gyear<C> (), xsd),
          &serializer_impl<gyear>,
          false);

        typedef gmonth_day<C, simple_type> gmonth_day;
        register_type (
          typeid (gmonth_day),
          qualified_name (bits::gmonth_day<C> (), xsd),
          &serializer_impl<gmonth_day>,
          false);

        typedef gyear_month<C, simple_type> gyear_month;
        register_type (
          typeid (gyear_month),
          qualified_name (bits::gyear_month<C> (), xsd),
          &serializer_impl<gyear_month>,
          false);

        typedef date<C, simple_type> date;
        register_type (
          typeid (date),
          qualified_name (bits::date<C> (), xsd),
          &serializer_impl<date>,
          false);

        typedef time<C, simple_type> time;
        register_type (
          typeid (time),
          qualified_name (bits::time<C> (), xsd),
          &serializer_impl<time>,
          false);

        typedef date_time<C, simple_type> date_time;
        register_type (
          typeid (date_time),
          qualified_name (bits::date_time<C> (), xsd),
          &serializer_impl<date_time>,
          false);

        typedef duration<C, simple_type> duration;
        register_type (
          typeid (duration),
          qualified_name (bits::duration<C> (), xsd),
          &serializer_impl<duration>,
          false);


        // Entity.
        //
        typedef entity<C, ncname> entity;
        register_type (
          typeid (entity),
          qualified_name (bits::entity<C> (), xsd),
          &serializer_impl<entity>,
          false);

        typedef entities<C, simple_type, entity> entities;
        register_type (
          typeid (entities),
          qualified_name (bits::entities<C> (), xsd),
          &serializer_impl<entities>,
          false);
      }

      template <typename C>
      void type_serializer_map<C>::
      register_type (const type_id& tid,
                     const qualified_name& name,
                     serializer s,
                     bool override)
      {
        if (override || type_map_.find (&tid) == type_map_.end ())
          type_map_[&tid] = type_info (name, s);
      }

      template <typename C>
      void type_serializer_map<C>::
      unregister_type (const type_id& tid)
      {
        type_map_.erase (&tid);
      }

      template <typename C>
      void type_serializer_map<C>::
      register_element (const qualified_name& root,
                        const qualified_name& subst,
                        const type_id& tid,
                        serializer s)
      {
        element_map_[root][&tid] = type_info (subst, s);
      }

      template <typename C>
      void type_serializer_map<C>::
      unregister_element (const qualified_name& root, const type_id& tid)
      {
        typename element_map::iterator i (element_map_.find (root));

        if (i != element_map_.end ())
        {
          i->second.erase (&tid);

          if (i->second.empty ())
            element_map_.erase (root);
        }
      }

      template <typename C>
      void type_serializer_map<C>::
      serialize (const C* name, // element name
                 const C* ns,   // element namespace
                 bool global,
                 bool qualified,
                 xercesc::DOMElement& parent,
                 const type& x) const
      {
        const type_id& tid (typeid (x));

        // First see if we can find a substitution.
        //
        if (global)
        {
          typename element_map::const_iterator i (
            element_map_.find (qualified_name (name, ns)));

          if (i != element_map_.end ())
          {
            if (const type_info* ti = find_substitution (i->second, tid))
            {
              xercesc::DOMElement& e (
                xml::dom::create_element (
                  ti->name ().name ().c_str (),
                  ti->name ().namespace_ ().c_str (),
                  parent));

              ti->serializer () (e, x);
              return;
            }
          }
        }

        // The last resort is xsi:type.
        //
        if (const type_info* ti = find (tid))
        {
          xercesc::DOMElement& e (
            qualified
            ? xml::dom::create_element (name, ns, parent)
            : xml::dom::create_element (name, parent));

          ti->serializer () (e, x);
          set_xsi_type (parent, e, *ti);
          return;
        }

        throw no_type_info<C> (std::basic_string<C> (),
                               std::basic_string<C> ()); //@@ TODO
      }

      template <typename C>
      void type_serializer_map<C>::
      serialize (const C* static_name,
                 const C* static_ns,
                 xercesc::DOMElement& e,
                 const qualified_name& qn,
                 const type& x) const
      {
        const type_id& tid (typeid (x));

        // First see if this is a substitution.
        //
        if (qn.name () != static_name || qn.namespace_ () != static_ns)
        {
          typename element_map::const_iterator i (
            element_map_.find (qualified_name (static_name, static_ns)));

          if (i != element_map_.end ())
          {
            if (const type_info* ti = find_substitution (i->second, tid))
            {
              if (ti->name ().name () != qn.name () ||
                  ti->name ().namespace_ () != qn.namespace_ ())
              {
                throw unexpected_element<C> (
                  qn.name (), qn.namespace_ (),
                  ti->name ().name (), ti->name ().namespace_ ());
              }

              ti->serializer () (e, x);
              return;
            }
          }

          // This is not a valid substitution.
          //
          throw unexpected_element<C> (qn.name (), qn.namespace_ (),
                                       static_name, static_ns);
        }

        // The last resort is xsi:type.
        //
        if (const type_info* ti = find (tid))
        {
          ti->serializer () (e, x);
          set_xsi_type (e, e, *ti);
          return;
        }

        throw no_type_info<C> (std::basic_string<C> (),
                               std::basic_string<C> ()); //@@ TODO
      }

      template <typename C>
      xml::dom::auto_ptr<xercesc::DOMDocument> type_serializer_map<C>::
      serialize (const C* name,
                 const C* ns,
                 const xml::dom::namespace_infomap<C>& m,
                 const type& x,
                 unsigned long flags) const
      {
        const type_id& tid (typeid (x));

        // See if we can find a substitution.
        //
        {
          typename element_map::const_iterator i (
            element_map_.find (qualified_name (name, ns)));

          if (i != element_map_.end ())
          {
            if (const type_info* ti = find_substitution (i->second, tid))
            {
              return xml::dom::serialize<C> (
                ti->name ().name (), ti->name ().namespace_ (), m, flags);
            }
          }
        }

        // If there is no substitution then serialize() will have to
        // find suitable xsi:type.
        //
        return xml::dom::serialize<C> (name, ns, m, flags);
      }

      template <typename C>
      const typename type_serializer_map<C>::type_info*
      type_serializer_map<C>::
      find (const type_id& tid) const
      {
        typename type_map::const_iterator i (type_map_.find (&tid));
        return i == type_map_.end () ? 0 : &i->second;
      }

      template <typename C>
      const typename type_serializer_map<C>::type_info*
      type_serializer_map<C>::
      find_substitution (const subst_map& start, const type_id& tid) const
      {
        typename subst_map::const_iterator i (start.find (&tid));

        if (i != start.end ())
          return &i->second;
        else
        {
          for (i = start.begin (); i != start.end (); ++i)
          {
            typename element_map::const_iterator j (
              element_map_.find (i->second.name ()));

            if (j != element_map_.end ())
            {
              if (const type_info* ti = find_substitution (j->second, tid))
                return ti;
            }
          }
        }

        return 0;
      }

      template <typename C>
      void type_serializer_map<C>::
      set_xsi_type (xercesc::DOMElement& parent,
                    xercesc::DOMElement& e,
                    const type_info& ti) const
      {
        std::basic_string<C> id;
        const std::basic_string<C>& ns (ti.name ().namespace_ ());

        if (!ns.empty ())
        {
          id = xml::dom::prefix (ns, e);

          if (!id.empty ())
            id += C (':');
        }

        id += ti.name ().name ();

        std::basic_string<C> name = xml::dom::prefix (
          xml::bits::xsi_namespace<C> (), parent, xml::bits::xsi_prefix<C> ());

        if (!name.empty ())
          name += C (':');

        name += xml::bits::type<C> ();

        e.setAttributeNS (
          xercesc::SchemaSymbols::fgURI_XSI,
          xml::string (name).c_str (),
          xml::string (id).c_str ());
      }


      // type_serializer_plate
      //
      template<unsigned long id, typename C>
      type_serializer_plate<id, C>::
      type_serializer_plate ()
      {
        if (count == 0)
          map = new type_serializer_map<C>;

        ++count;
      }

      template<unsigned long id, typename C>
      type_serializer_plate<id, C>::
      ~type_serializer_plate ()
      {
        if (--count == 0)
          delete map;
      }


      //
      //
      template<typename T>
      void
      serializer_impl (xercesc::DOMElement& e, const type& x)
      {
        e << static_cast<const T&> (x);
      }

      // type_serializer_initializer
      //
      template<unsigned long id, typename C, typename T>
      type_serializer_initializer<id, C, T>::
      type_serializer_initializer (const C* name, const C* ns)
      {
        type_serializer_map_instance<id, C> ().register_type (
          typeid (T),
          xml::qualified_name<C> (name, ns),
          &serializer_impl<T>);
      }

      template<unsigned long id, typename C, typename T>
      type_serializer_initializer<id, C, T>::
      ~type_serializer_initializer ()
      {
        type_serializer_map_instance<id, C> ().unregister_type (typeid (T));
      }

      // element_serializer_initializer
      //
      template<unsigned long id, typename C, typename T>
      element_serializer_initializer<id, C, T>::
      element_serializer_initializer (const C* root_name, const C* root_ns,
                                   const C* subst_name, const C* subst_ns)
          : root_name_ (root_name), root_ns_ (root_ns)
      {
        type_serializer_map_instance<id, C> ().register_element (
          xml::qualified_name<C> (root_name, root_ns),
          xml::qualified_name<C> (subst_name, subst_ns),
          typeid (T),
          &serializer_impl<T>);
      }

      template<unsigned long id, typename C, typename T>
      element_serializer_initializer<id, C, T>::
      ~element_serializer_initializer ()
      {
        type_serializer_map_instance<id, C> ().unregister_element (
          xml::qualified_name<C> (root_name_, root_ns_), typeid (T));
      }
    }
  }
}
