// file      : xsd/cxx/tree/type-factory-map.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <xercesc/validators/schema/SchemaSymbols.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.hxx>        // xml::{string, transcode}
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/elements.hxx>      // xml::{prefix, uq_name}
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/bits/literals.hxx> // xml::bits::{xml_namespace, etc}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/bits/literals.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // type_factory_map
      //
      template <typename C>
      type_factory_map<C>::
      type_factory_map ()
      {
        // Register factories for default instantiations of built-in,
        // non-fundamental types.
        //

        std::basic_string<C> xsd (bits::xml_schema<C> ());


        // anyType and anySimpleType.
        //
        register_type (
          qualified_name (bits::any_type<C> (), xsd),
          &factory_impl<type>,
          false);

        typedef simple_type<type> simple_type;
        register_type (
          qualified_name (bits::any_simple_type<C> (), xsd),
          &factory_impl<simple_type>,
          false);


        // Strings
        //
        typedef string<C, simple_type> string;
        register_type (
          qualified_name (bits::string<C> (), xsd),
          &factory_impl<string>,
          false);

        typedef normalized_string<C, string> normalized_string;
        register_type (
          qualified_name (bits::normalized_string<C> (), xsd),
          &factory_impl<normalized_string>,
          false);

        typedef token<C, normalized_string> token;
        register_type (
          qualified_name (bits::token<C> (), xsd),
          &factory_impl<token>,
          false);

        typedef name<C, token> name;
        register_type (
          qualified_name (bits::name<C> (), xsd),
          &factory_impl<name>,
          false);

        typedef nmtoken<C, token> nmtoken;
        register_type (
          qualified_name (bits::nmtoken<C> (), xsd),
          &factory_impl<nmtoken>,
          false);

        typedef nmtokens<C, simple_type, nmtoken> nmtokens;
        register_type (
          qualified_name (bits::nmtokens<C> (), xsd),
          &factory_impl<nmtokens>,
          false);

        typedef ncname<C, name> ncname;
        register_type (
          qualified_name (bits::ncname<C> (), xsd),
          &factory_impl<ncname>,
          false);

        typedef language<C, token> language;
        register_type (
          qualified_name (bits::language<C> (), xsd),
          &factory_impl<language>,
          false);


        // ID/IDREF.
        //
        typedef id<C, ncname> id;
        register_type (
          qualified_name (bits::id<C> (), xsd),
          &factory_impl<id>,
          false);

        typedef idref<C, ncname, type> idref;
        register_type (
          qualified_name (bits::idref<C> (), xsd),
          &factory_impl<idref>,
          false);

        typedef idrefs<C, simple_type, idref> idrefs;
        register_type (
          qualified_name (bits::idrefs<C> (), xsd),
          &factory_impl<idrefs>,
          false);


        // URI.
        //
        typedef uri<C, simple_type> uri;
        register_type (
          qualified_name (bits::any_uri<C> (), xsd),
          &factory_impl<uri>,
          false);


        // Qualified name.
        //
        typedef qname<C, simple_type, uri, ncname> qname;
        register_type (
          qualified_name (bits::qname<C> (), xsd),
          &factory_impl<qname>,
          false);


        // Binary.
        //
        typedef base64_binary<C, simple_type> base64_binary;
        register_type (
          qualified_name (bits::base64_binary<C> (), xsd),
          &factory_impl<base64_binary>,
          false);

        typedef hex_binary<C, simple_type> hex_binary;
        register_type (
          qualified_name (bits::hex_binary<C> (), xsd),
          &factory_impl<hex_binary>,
          false);


        // Date/time.
        //
        typedef gday<C, simple_type> gday;
        register_type (
          qualified_name (bits::gday<C> (), xsd),
          &factory_impl<gday>,
          false);

        typedef gmonth<C, simple_type> gmonth;
        register_type (
          qualified_name (bits::gmonth<C> (), xsd),
          &factory_impl<gmonth>,
          false);

        typedef gyear<C, simple_type> gyear;
        register_type (
          qualified_name (bits::gyear<C> (), xsd),
          &factory_impl<gyear>,
          false);

        typedef gmonth_day<C, simple_type> gmonth_day;
        register_type (
          qualified_name (bits::gmonth_day<C> (), xsd),
          &factory_impl<gmonth_day>,
          false);

        typedef gyear_month<C, simple_type> gyear_month;
        register_type (
          qualified_name (bits::gyear_month<C> (), xsd),
          &factory_impl<gyear_month>,
          false);

        typedef date<C, simple_type> date;
        register_type (
          qualified_name (bits::date<C> (), xsd),
          &factory_impl<date>,
          false);

        typedef time<C, simple_type> time;
        register_type (
          qualified_name (bits::time<C> (), xsd),
          &factory_impl<time>,
          false);

        typedef date_time<C, simple_type> date_time;
        register_type (
          qualified_name (bits::date_time<C> (), xsd),
          &factory_impl<date_time>,
          false);

        typedef duration<C, simple_type> duration;
        register_type (
          qualified_name (bits::duration<C> (), xsd),
          &factory_impl<duration>,
          false);


        // Entity.
        //
        typedef entity<C, ncname> entity;
        register_type (
          qualified_name (bits::entity<C> (), xsd),
          &factory_impl<entity>,
          false);

        typedef entities<C, simple_type, entity> entities;
        register_type (
          qualified_name (bits::entities<C> (), xsd),
          &factory_impl<entities>,
          false);
      }

      template <typename C>
      void type_factory_map<C>::
      register_type (const qualified_name& name,
                     factory f,
                     bool override)
      {
        if (override || type_map_.find (name) == type_map_.end ())
          type_map_[name] = f;
      }

      template <typename C>
      void type_factory_map<C>::
      unregister_type (const qualified_name& name)
      {
        type_map_.erase (name);
      }

      template <typename C>
      void type_factory_map<C>::
      register_element (const qualified_name& root,
                        const qualified_name& subst,
                        factory f)
      {
        element_map_[root][subst] = f;
      }

      template <typename C>
      void type_factory_map<C>::
      unregister_element (const qualified_name& root,
                          const qualified_name& subst)
      {
        typename element_map::iterator i (element_map_.find (root));

        if (i != element_map_.end ())
        {
          i->second.erase (subst);

          if (i->second.empty ())
            element_map_.erase (i);
        }
      }

      template <typename C>
      typename type_factory_map<C>::factory type_factory_map<C>::
      find (const qualified_name& name) const
      {
        typename type_map::const_iterator i (type_map_.find (name));
        return i == type_map_.end () ? 0 : i->second;
      }

      template <typename C>
      std::auto_ptr<type> type_factory_map<C>::
      create (const C* name,
              const C* ns,
              factory static_type,
              bool global,
              bool qualified,
              const xercesc::DOMElement& e,
              const qualified_name& qn,
              tree::flags flags,
              container* c) const
      {
        factory f = 0;

        // See if we've got a straight match.
        //
        if (qn.name () == name &&
            (qualified ? qn.namespace_ () == ns : ns[0] == C ('\0')))
        {
          f = static_type;
        }
        else if (global)
        {
          // See if we have a substitution.
          //
          typename element_map::const_iterator i (
            element_map_.find (qualified_name (name, ns)));

          if (i != element_map_.end ())
          {
            f = find_substitution (i->second, qn);
          }
        }

        if (f == 0)
          return std::auto_ptr<type> (0); // No match.

        // Check for xsi:type
        //
        {
          const XMLCh* v (
            e.getAttributeNS (
              xercesc::SchemaSymbols::fgURI_XSI,
              xercesc::SchemaSymbols::fgXSI_TYPE));

          if (v != 0 && v[0] != XMLCh (0))
            f = find_type (xml::transcode<C> (v), e);
        }

        return f (e, flags, c);
      }

      template <typename C>
      template <typename T>
      std::auto_ptr<type> type_factory_map<C>::
      traits_adapter (const xercesc::DOMElement& e, flags f, container* c)
      {
        std::auto_ptr<T> r (traits<T, C>::create (e, f, c));
        return std::auto_ptr<type> (r.release ());
      }

      template <typename C>
      typename type_factory_map<C>::factory type_factory_map<C>::
      find_substitution (const subst_map& start,
                         const qualified_name& name) const
      {
        typename subst_map::const_iterator i (start.find (name));

        if (i != start.end ())
          return i->second;
        else
        {
          for (i = start.begin (); i != start.end (); ++i)
          {
            typename element_map::const_iterator j (
              element_map_.find (i->first));

            if (j != element_map_.end ())
            {
              if (factory f = find_substitution (j->second, name))
                return f;
            }
          }
        }

        return 0;
      }

      template <typename C>
      typename type_factory_map<C>::factory type_factory_map<C>::
      find_type (const std::basic_string<C>& name,
                 const xercesc::DOMElement& e) const
      {
        using std::basic_string;

        basic_string<C> ns_name, uq_name (xml::uq_name (name));

        // Copied with modifications from xml/dom/elements.hxx.
        //
        std::basic_string<C> p (xml::prefix (name));

        // 'xml' prefix requires special handling and Xerces folks refuse
        // to handle this in DOM so I have to do it myself.
        //
        if (p == xml::bits::xml_prefix<C> ())
          ns_name = xml::bits::xml_namespace<C> ();
        else
        {
          const XMLCh* xns (
            e.lookupNamespaceURI (
              p.empty () ? 0 : xml::string (p).c_str ()));

          if (xns != 0)
            ns_name = xml::transcode<C> (xns);
          else
          {
            // See if we've got any no-namespace types.
            //
            if (!p.empty ())
              throw no_prefix_mapping<C> (p);
          }
        }

        factory f (find (qualified_name (uq_name, ns_name)));

        if (f == 0)
          throw no_type_info<C> (uq_name, ns_name);

        return f;
      }


      // type_factory_plate
      //
      template<unsigned long id, typename C>
      type_factory_plate<id, C>::
      type_factory_plate ()
      {
        if (count == 0)
          map = new type_factory_map<C>;

        ++count;
      }

      template<unsigned long id, typename C>
      type_factory_plate<id, C>::
      ~type_factory_plate ()
      {
        if (--count == 0)
          delete map;
      }


      //
      //
      template<typename T>
      std::auto_ptr<type>
      factory_impl (const xercesc::DOMElement& e, flags f, container* c)
      {
        return std::auto_ptr<type> (new T (e, f, c));
      }

      //
      //
      template<unsigned long id, typename C, typename T>
      type_factory_initializer<id, C, T>::
      type_factory_initializer (const C* name, const C* ns)
          : name_ (name), ns_ (ns)
      {
        type_factory_map_instance<id, C> ().register_type (
          xml::qualified_name<C> (name, ns), &factory_impl<T>);
      }

      template<unsigned long id, typename C, typename T>
      type_factory_initializer<id, C, T>::
      ~type_factory_initializer ()
      {
        type_factory_map_instance<id, C> ().unregister_type (
          xml::qualified_name<C> (name_, ns_));
      }

      //
      //
      template<unsigned long id, typename C, typename T>
      element_factory_initializer<id, C, T>::
      element_factory_initializer (const C* root_name, const C* root_ns,
                                const C* subst_name, const C* subst_ns)
          : root_name_ (root_name), root_ns_ (root_ns),
            subst_name_ (subst_name), subst_ns_ (subst_ns)
      {
        type_factory_map_instance<id, C> ().register_element (
          xml::qualified_name<C> (root_name, root_ns),
          xml::qualified_name<C> (subst_name, subst_ns),
          &factory_impl<T>);
      }

      template<unsigned long id, typename C, typename T>
      element_factory_initializer<id, C, T>::
      ~element_factory_initializer ()
      {
        type_factory_map_instance<id, C> ().unregister_element (
          xml::qualified_name<C> (root_name_, root_ns_),
          xml::qualified_name<C> (subst_name_, subst_ns_));
      }
    }
  }
}
