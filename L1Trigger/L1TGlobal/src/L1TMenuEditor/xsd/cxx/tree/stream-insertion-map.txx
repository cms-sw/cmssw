// file      : xsd/cxx/tree/stream-insertion-map.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/stream-insertion.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/bits/literals.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // stream_insertion_map
      //
      template <typename S, typename C>
      stream_insertion_map<S, C>::
      stream_insertion_map ()
      {
        // Register inserters for built-in non-fundamental types.
        //
        std::basic_string<C> xsd (bits::xml_schema<C> ());


        // anyType and anySimpleType.
        //
        register_type (
          typeid (type),
          qualified_name (bits::any_type<C> (), xsd),
          &inserter_impl<S, type>,
          false);

        typedef simple_type<type> simple_type;
        register_type (
          typeid (simple_type),
          qualified_name (bits::any_simple_type<C> (), xsd),
          &inserter_impl<S, simple_type>,
          false);


        // Strings
        //
        typedef string<C, simple_type> string;
        register_type (
          typeid (string),
          qualified_name (bits::string<C> (), xsd),
          &inserter_impl<S, string>,
          false);

        typedef normalized_string<C, string> normalized_string;
        register_type (
          typeid (normalized_string),
          qualified_name (bits::normalized_string<C> (), xsd),
          &inserter_impl<S, normalized_string>,
          false);

        typedef token<C, normalized_string> token;
        register_type (
          typeid (token),
          qualified_name (bits::token<C> (), xsd),
          &inserter_impl<S, token>,
          false);

        typedef name<C, token> name;
        register_type (
          typeid (name),
          qualified_name (bits::name<C> (), xsd),
          &inserter_impl<S, name>,
          false);

        typedef nmtoken<C, token> nmtoken;
        register_type (
          typeid (nmtoken),
          qualified_name (bits::nmtoken<C> (), xsd),
          &inserter_impl<S, nmtoken>,
          false);

        typedef nmtokens<C, simple_type, nmtoken> nmtokens;
        register_type (
          typeid (nmtokens),
          qualified_name (bits::nmtokens<C> (), xsd),
          &inserter_impl<S, nmtokens>,
          false);

        typedef ncname<C, name> ncname;
        register_type (
          typeid (ncname),
          qualified_name (bits::ncname<C> (), xsd),
          &inserter_impl<S, ncname>,
          false);

        typedef language<C, token> language;
        register_type (
          typeid (language),
          qualified_name (bits::language<C> (), xsd),
          &inserter_impl<S, language>,
          false);


        // ID/IDREF.
        //
        typedef id<C, ncname> id;
        register_type (
          typeid (id),
          qualified_name (bits::id<C> (), xsd),
          &inserter_impl<S, id>,
          false);

        typedef idref<C, ncname, type> idref;
        register_type (
          typeid (idref),
          qualified_name (bits::idref<C> (), xsd),
          &inserter_impl<S, idref>,
          false);

        typedef idrefs<C, simple_type, idref> idrefs;
        register_type (
          typeid (idrefs),
          qualified_name (bits::idrefs<C> (), xsd),
          &inserter_impl<S, idrefs>,
          false);


        // URI.
        //
        typedef uri<C, simple_type> uri;
        register_type (
          typeid (uri),
          qualified_name (bits::any_uri<C> (), xsd),
          &inserter_impl<S, uri>,
          false);


        // Qualified name.
        //
        typedef qname<C, simple_type, uri, ncname> qname;
        register_type (
          typeid (qname),
          qualified_name (bits::qname<C> (), xsd),
          &inserter_impl<S, qname>,
          false);


        // Binary.
        //
        typedef base64_binary<C, simple_type> base64_binary;
        register_type (
          typeid (base64_binary),
          qualified_name (bits::base64_binary<C> (), xsd),
          &inserter_impl<S, base64_binary>,
          false);

        typedef hex_binary<C, simple_type> hex_binary;
        register_type (
          typeid (hex_binary),
          qualified_name (bits::hex_binary<C> (), xsd),
          &inserter_impl<S, hex_binary>,
          false);


        // Date/time.
        //
        typedef gday<C, simple_type> gday;
        register_type (
          typeid (gday),
          qualified_name (bits::gday<C> (), xsd),
          &inserter_impl<S, gday>,
          false);

        typedef gmonth<C, simple_type> gmonth;
        register_type (
          typeid (gmonth),
          qualified_name (bits::gmonth<C> (), xsd),
          &inserter_impl<S, gmonth>,
          false);

        typedef gyear<C, simple_type> gyear;
        register_type (
          typeid (gyear),
          qualified_name (bits::gyear<C> (), xsd),
          &inserter_impl<S, gyear>,
          false);

        typedef gmonth_day<C, simple_type> gmonth_day;
        register_type (
          typeid (gmonth_day),
          qualified_name (bits::gmonth_day<C> (), xsd),
          &inserter_impl<S, gmonth_day>,
          false);

        typedef gyear_month<C, simple_type> gyear_month;
        register_type (
          typeid (gyear_month),
          qualified_name (bits::gyear_month<C> (), xsd),
          &inserter_impl<S, gyear_month>,
          false);

        typedef date<C, simple_type> date;
        register_type (
          typeid (date),
          qualified_name (bits::date<C> (), xsd),
          &inserter_impl<S, date>,
          false);

        typedef time<C, simple_type> time;
        register_type (
          typeid (time),
          qualified_name (bits::time<C> (), xsd),
          &inserter_impl<S, time>,
          false);

        typedef date_time<C, simple_type> date_time;
        register_type (
          typeid (date_time),
          qualified_name (bits::date_time<C> (), xsd),
          &inserter_impl<S, date_time>,
          false);

        typedef duration<C, simple_type> duration;
        register_type (
          typeid (duration),
          qualified_name (bits::duration<C> (), xsd),
          &inserter_impl<S, duration>,
          false);


        // Entity.
        //
        typedef entity<C, ncname> entity;
        register_type (
          typeid (entity),
          qualified_name (bits::entity<C> (), xsd),
          &inserter_impl<S, entity>,
          false);

        typedef entities<C, simple_type, entity> entities;
        register_type (
          typeid (entities),
          qualified_name (bits::entities<C> (), xsd),
          &inserter_impl<S, entities>,
          false);
      }

      template <typename S, typename C>
      void stream_insertion_map<S, C>::
      register_type (const type_id& tid,
                     const qualified_name& name,
                     inserter i,
                     bool override)
      {
        if (override || type_map_.find (&tid) == type_map_.end ())
          type_map_[&tid] = type_info (name, i);
      }

      template <typename S, typename C>
      void stream_insertion_map<S, C>::
      unregister_type (const type_id& tid)
      {
        type_map_.erase (&tid);
      }

      template <typename S, typename C>
      void stream_insertion_map<S, C>::
      insert (ostream<S>& s, const type& x)
      {
        if (const type_info* ti = find (typeid (x)))
        {
          const qualified_name& qn (ti->name ());

          s << qn.namespace_ () << qn.name ();
          ti->inserter () (s, x);
        }
        else
          throw no_type_info<C> (std::basic_string<C> (),
                                 std::basic_string<C> ()); // @@ TODO
      }

      template <typename S, typename C>
      const typename stream_insertion_map<S, C>::type_info*
      stream_insertion_map<S, C>::
      find (const type_id& tid) const
      {
        typename type_map::const_iterator i (type_map_.find (&tid));
        return i == type_map_.end () ? 0 : &i->second;
      }


      // stream_insertion_plate
      //
      template<unsigned long id, typename S, typename C>
      stream_insertion_plate<id, S, C>::
      stream_insertion_plate ()
      {
        if (count == 0)
          map = new stream_insertion_map<S, C>;

        ++count;
      }

      template<unsigned long id, typename S, typename C>
      stream_insertion_plate<id, S, C>::
      ~stream_insertion_plate ()
      {
        if (--count == 0)
          delete map;
      }

      //
      //
      template<typename S, typename T>
      void
      inserter_impl (ostream<S>& s, const type& x)
      {
        s << static_cast<const T&> (x);
      }

      // stream_insertion_initializer
      //
      template<unsigned long id, typename S, typename C, typename T>
      stream_insertion_initializer<id, S, C, T>::
      stream_insertion_initializer (const C* name, const C* ns)
      {
        stream_insertion_map_instance<id, S, C> ().register_type (
          typeid (T),
          xml::qualified_name<C> (name, ns),
          &inserter_impl<S, T>);
      }

      template<unsigned long id, typename S, typename C, typename T>
      stream_insertion_initializer<id, S, C, T>::
      ~stream_insertion_initializer ()
      {
        stream_insertion_map_instance<id, S, C> ().unregister_type (
          typeid (T));
      }
    }
  }
}
