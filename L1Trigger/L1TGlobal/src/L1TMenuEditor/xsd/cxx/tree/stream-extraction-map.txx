// file      : xsd/cxx/tree/stream-extraction-map.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/stream-extraction.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/bits/literals.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // stream_extraction_map
      //
      template <typename S, typename C>
      stream_extraction_map<S, C>::
      stream_extraction_map ()
      {
        // Register extractors for built-in non-fundamental types.
        //
        std::basic_string<C> xsd (bits::xml_schema<C> ());


        // anyType and anySimpleType.
        //
        register_type (
          qualified_name (bits::any_type<C> (), xsd),
          &extractor_impl<S, type>,
          false);

        typedef simple_type<type> simple_type;
        register_type (
          qualified_name (bits::any_simple_type<C> (), xsd),
          &extractor_impl<S, simple_type>,
          false);


        // Strings
        //
        typedef string<C, simple_type> string;
        register_type (
          qualified_name (bits::string<C> (), xsd),
          &extractor_impl<S, string>,
          false);

        typedef normalized_string<C, string> normalized_string;
        register_type (
          qualified_name (bits::normalized_string<C> (), xsd),
          &extractor_impl<S, normalized_string>,
          false);

        typedef token<C, normalized_string> token;
        register_type (
          qualified_name (bits::token<C> (), xsd),
          &extractor_impl<S, token>,
          false);

        typedef name<C, token> name;
        register_type (
          qualified_name (bits::name<C> (), xsd),
          &extractor_impl<S, name>,
          false);

        typedef nmtoken<C, token> nmtoken;
        register_type (
          qualified_name (bits::nmtoken<C> (), xsd),
          &extractor_impl<S, nmtoken>,
          false);

        typedef nmtokens<C, simple_type, nmtoken> nmtokens;
        register_type (
          qualified_name (bits::nmtokens<C> (), xsd),
          &extractor_impl<S, nmtokens>,
          false);

        typedef ncname<C, name> ncname;
        register_type (
          qualified_name (bits::ncname<C> (), xsd),
          &extractor_impl<S, ncname>,
          false);

        typedef language<C, token> language;
        register_type (
          qualified_name (bits::language<C> (), xsd),
          &extractor_impl<S, language>,
          false);


        // ID/IDREF.
        //
        typedef id<C, ncname> id;
        register_type (
          qualified_name (bits::id<C> (), xsd),
          &extractor_impl<S, id>,
          false);

        typedef idref<C, ncname, type> idref;
        register_type (
          qualified_name (bits::idref<C> (), xsd),
          &extractor_impl<S, idref>,
          false);

        typedef idrefs<C, simple_type, idref> idrefs;
        register_type (
          qualified_name (bits::idrefs<C> (), xsd),
          &extractor_impl<S, idrefs>,
          false);


        // URI.
        //
        typedef uri<C, simple_type> uri;
        register_type (
          qualified_name (bits::any_uri<C> (), xsd),
          &extractor_impl<S, uri>,
          false);


        // Qualified name.
        //
        typedef qname<C, simple_type, uri, ncname> qname;
        register_type (
          qualified_name (bits::qname<C> (), xsd),
          &extractor_impl<S, qname>,
          false);


        // Binary.
        //
        typedef base64_binary<C, simple_type> base64_binary;
        register_type (
          qualified_name (bits::base64_binary<C> (), xsd),
          &extractor_impl<S, base64_binary>,
          false);

        typedef hex_binary<C, simple_type> hex_binary;
        register_type (
          qualified_name (bits::hex_binary<C> (), xsd),
          &extractor_impl<S, hex_binary>,
          false);


        // Date/time.
        //
        typedef gday<C, simple_type> gday;
        register_type (
          qualified_name (bits::gday<C> (), xsd),
          &extractor_impl<S, gday>,
          false);

        typedef gmonth<C, simple_type> gmonth;
        register_type (
          qualified_name (bits::gmonth<C> (), xsd),
          &extractor_impl<S, gmonth>,
          false);

        typedef gyear<C, simple_type> gyear;
        register_type (
          qualified_name (bits::gyear<C> (), xsd),
          &extractor_impl<S, gyear>,
          false);

        typedef gmonth_day<C, simple_type> gmonth_day;
        register_type (
          qualified_name (bits::gmonth_day<C> (), xsd),
          &extractor_impl<S, gmonth_day>,
          false);

        typedef gyear_month<C, simple_type> gyear_month;
        register_type (
          qualified_name (bits::gyear_month<C> (), xsd),
          &extractor_impl<S, gyear_month>,
          false);

        typedef date<C, simple_type> date;
        register_type (
          qualified_name (bits::date<C> (), xsd),
          &extractor_impl<S, date>,
          false);

        typedef time<C, simple_type> time;
        register_type (
          qualified_name (bits::time<C> (), xsd),
          &extractor_impl<S, time>,
          false);

        typedef date_time<C, simple_type> date_time;
        register_type (
          qualified_name (bits::date_time<C> (), xsd),
          &extractor_impl<S, date_time>,
          false);

        typedef duration<C, simple_type> duration;
        register_type (
          qualified_name (bits::duration<C> (), xsd),
          &extractor_impl<S, duration>,
          false);


        // Entity.
        //
        typedef entity<C, ncname> entity;
        register_type (
          qualified_name (bits::entity<C> (), xsd),
          &extractor_impl<S, entity>,
          false);

        typedef entities<C, simple_type, entity> entities;
        register_type (
          qualified_name (bits::entities<C> (), xsd),
          &extractor_impl<S, entities>,
          false);
      }

      template <typename S, typename C>
      void stream_extraction_map<S, C>::
      register_type (const qualified_name& name,
                     extractor e,
                     bool override)
      {
        if (override || type_map_.find (name) == type_map_.end ())
          type_map_[name] = e;
      }

      template <typename S, typename C>
      void stream_extraction_map<S, C>::
      unregister_type (const qualified_name& name)
      {
        type_map_.erase (name);
      }

      template <typename S, typename C>
      std::auto_ptr<type> stream_extraction_map<S, C>::
      extract (istream<S>& s, flags f, container* c)
      {
        std::basic_string<C> name, ns;
        s >> ns >> name;

        if (extractor e = find (qualified_name (name, ns)))
        {
          return e (s, f, c);
        }
        else
          throw no_type_info<C> (name, ns);
      }

      template <typename S, typename C>
      typename stream_extraction_map<S, C>::extractor
      stream_extraction_map<S, C>::
      find (const qualified_name& name) const
      {
        typename type_map::const_iterator i (type_map_.find (name));
        return i == type_map_.end () ? 0 : i->second;
      }


      // stream_extraction_plate
      //
      template<unsigned long id, typename S, typename C>
      stream_extraction_plate<id, S, C>::
      stream_extraction_plate ()
      {
        if (count == 0)
          map = new stream_extraction_map<S, C>;

        ++count;
      }

      template<unsigned long id, typename S, typename C>
      stream_extraction_plate<id, S, C>::
      ~stream_extraction_plate ()
      {
        if (--count == 0)
          delete map;
      }

      //
      //
      template<typename S, typename T>
      std::auto_ptr<type>
      extractor_impl (istream<S>& s, flags f, container* c)
      {
        return std::auto_ptr<type> (new T (s, f, c));
      }


      // stream_extraction_initializer
      //
      template<unsigned long id, typename S, typename C, typename T>
      stream_extraction_initializer<id, S, C, T>::
      stream_extraction_initializer (const C* name, const C* ns)
          : name_ (name), ns_ (ns)
      {
        stream_extraction_map_instance<id, S, C> ().register_type (
          xml::qualified_name<C> (name, ns), &extractor_impl<S, T>);
      }

      template<unsigned long id, typename S, typename C, typename T>
      stream_extraction_initializer<id, S, C, T>::
      ~stream_extraction_initializer ()
      {
        stream_extraction_map_instance<id, S, C> ().unregister_type (
          xml::qualified_name<C> (name_, ns_));
      }
    }
  }
}
