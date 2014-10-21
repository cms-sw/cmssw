// file      : xsd/cxx/tree/std-ostream-map.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/std-ostream-operators.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // std_ostream_map
      //
      template <typename C>
      std_ostream_map<C>::
      std_ostream_map ()
      {
        // anyType and anySimpleType.
        //
        register_type (
          typeid (type),
          &inserter_impl<C, type>,
          false);

        typedef simple_type<type> simple_type;
        register_type (
          typeid (simple_type),
          &inserter_impl<C, simple_type>,
          false);


        // Strings
        //
        typedef string<C, simple_type> string;
        register_type (
          typeid (string),
          &inserter_impl<C, string>,
          false);

        typedef normalized_string<C, string> normalized_string;
        register_type (
          typeid (normalized_string),
          &inserter_impl<C, normalized_string>,
          false);

        typedef token<C, normalized_string> token;
        register_type (
          typeid (token),
          &inserter_impl<C, token>,
          false);

        typedef name<C, token> name;
        register_type (
          typeid (name),
          &inserter_impl<C, name>,
          false);

        typedef nmtoken<C, token> nmtoken;
        register_type (
          typeid (nmtoken),
          &inserter_impl<C, nmtoken>,
          false);

        typedef nmtokens<C, simple_type, nmtoken> nmtokens;
        register_type (
          typeid (nmtokens),
          &inserter_impl<C, nmtokens>,
          false);

        typedef ncname<C, name> ncname;
        register_type (
          typeid (ncname),
          &inserter_impl<C, ncname>,
          false);

        typedef language<C, token> language;
        register_type (
          typeid (language),
          &inserter_impl<C, language>,
          false);


        // ID/IDREF.
        //
        typedef id<C, ncname> id;
        register_type (
          typeid (id),
          &inserter_impl<C, id>,
          false);

        typedef idref<C, ncname, type> idref;
        register_type (
          typeid (idref),
          &inserter_impl<C, idref>,
          false);

        typedef idrefs<C, simple_type, idref> idrefs;
        register_type (
          typeid (idrefs),
          &inserter_impl<C, idrefs>,
          false);


        // URI.
        //
        typedef uri<C, simple_type> uri;
        register_type (
          typeid (uri),
          &inserter_impl<C, uri>,
          false);


        // Qualified name.
        //
        typedef qname<C, simple_type, uri, ncname> qname;
        register_type (
          typeid (qname),
          &inserter_impl<C, qname>,
          false);


        // Binary.
        //
        typedef base64_binary<C, simple_type> base64_binary;
        register_type (
          typeid (base64_binary),
          &inserter_impl<C, base64_binary>,
          false);

        typedef hex_binary<C, simple_type> hex_binary;
        register_type (
          typeid (hex_binary),
          &inserter_impl<C, hex_binary>,
          false);


        // Date/time.
        //
        typedef gday<C, simple_type> gday;
        register_type (
          typeid (gday),
          &inserter_impl<C, gday>,
          false);

        typedef gmonth<C, simple_type> gmonth;
        register_type (
          typeid (gmonth),
          &inserter_impl<C, gmonth>,
          false);

        typedef gyear<C, simple_type> gyear;
        register_type (
          typeid (gyear),
          &inserter_impl<C, gyear>,
          false);

        typedef gmonth_day<C, simple_type> gmonth_day;
        register_type (
          typeid (gmonth_day),
          &inserter_impl<C, gmonth_day>,
          false);

        typedef gyear_month<C, simple_type> gyear_month;
        register_type (
          typeid (gyear_month),
          &inserter_impl<C, gyear_month>,
          false);

        typedef date<C, simple_type> date;
        register_type (
          typeid (date),
          &inserter_impl<C, date>,
          false);

        typedef time<C, simple_type> time;
        register_type (
          typeid (time),
          &inserter_impl<C, time>,
          false);

        typedef date_time<C, simple_type> date_time;
        register_type (
          typeid (date_time),
          &inserter_impl<C, date_time>,
          false);

        typedef duration<C, simple_type> duration;
        register_type (
          typeid (duration),
          &inserter_impl<C, duration>,
          false);


        // Entity.
        //
        typedef entity<C, ncname> entity;
        register_type (
          typeid (entity),
          &inserter_impl<C, entity>,
          false);

        typedef entities<C, simple_type, entity> entities;
        register_type (
          typeid (entities),
          &inserter_impl<C, entities>,
          false);
      }

      template <typename C>
      void std_ostream_map<C>::
      register_type (const type_id& tid, inserter i, bool override)
      {
        if (override || type_map_.find (&tid) == type_map_.end ())
          type_map_[&tid] = i;
      }

      template <typename C>
      void std_ostream_map<C>::
      unregister_type (const type_id& tid)
      {
        type_map_.erase (&tid);
      }

      template <typename C>
      void std_ostream_map<C>::
      insert (std::basic_ostream<C>& os, const type& x)
      {
        if (inserter i = find (typeid (x)))
          i (os, x);
        else
          throw no_type_info<C> (std::basic_string<C> (),
                                 std::basic_string<C> ()); // @@ TODO
      }

      template <typename C>
      typename std_ostream_map<C>::inserter std_ostream_map<C>::
      find (const type_id& tid) const
      {
        typename type_map::const_iterator i (type_map_.find (&tid));
        return i == type_map_.end () ? 0 : i->second;
      }

      // std_ostream_plate
      //
      template<unsigned long id, typename C>
      std_ostream_plate<id, C>::
      std_ostream_plate ()
      {
        if (count == 0)
          map = new std_ostream_map<C>;

        ++count;
      }

      template<unsigned long id, typename C>
      std_ostream_plate<id, C>::
      ~std_ostream_plate ()
      {
        if (--count == 0)
          delete map;
      }

      //
      //
      template<typename C, typename T>
      void
      inserter_impl (std::basic_ostream<C>& os, const type& x)
      {
        os << static_cast<const T&> (x);
      }

      // std_ostream_initializer
      //
      template<unsigned long id, typename C, typename T>
      std_ostream_initializer<id, C, T>::
      std_ostream_initializer ()
      {
        std_ostream_map_instance<id, C> ().register_type (
          typeid (T), &inserter_impl<C, T>);
      }

      template<unsigned long id, typename C, typename T>
      std_ostream_initializer<id, C, T>::
      ~std_ostream_initializer ()
      {
        std_ostream_map_instance<id, C> ().unregister_type (typeid (T));
      }
    }
  }
}
