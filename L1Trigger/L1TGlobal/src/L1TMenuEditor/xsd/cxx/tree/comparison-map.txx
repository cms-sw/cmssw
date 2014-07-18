// file      : xsd/cxx/tree/comparison-map.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // comparison_map
      //
      template <typename C>
      comparison_map<C>::
      comparison_map ()
      {
        // anyType and anySimpleType.
        //

        //register_type (
        //  typeid (type),
        //  &comparator_impl<type>,
        //  false);

        typedef simple_type<type> simple_type;

        //register_type (
        //  typeid (simple_type),
        //  &comparator_impl<simple_type>,
        //  false);


        // Strings
        //
        typedef string<C, simple_type> string;
        register_type (
          typeid (string),
          &comparator_impl<string>,
          false);

        typedef normalized_string<C, string> normalized_string;
        register_type (
          typeid (normalized_string),
          &comparator_impl<normalized_string>,
          false);

        typedef token<C, normalized_string> token;
        register_type (
          typeid (token),
          &comparator_impl<token>,
          false);

        typedef name<C, token> name;
        register_type (
          typeid (name),
          &comparator_impl<name>,
          false);

        typedef nmtoken<C, token> nmtoken;
        register_type (
          typeid (nmtoken),
          &comparator_impl<nmtoken>,
          false);

        typedef nmtokens<C, simple_type, nmtoken> nmtokens;
        register_type (
          typeid (nmtokens),
          &comparator_impl<nmtokens>,
          false);

        typedef ncname<C, name> ncname;
        register_type (
          typeid (ncname),
          &comparator_impl<ncname>,
          false);

        typedef language<C, token> language;
        register_type (
          typeid (language),
          &comparator_impl<language>,
          false);


        // ID/IDREF.
        //
        typedef id<C, ncname> id;
        register_type (
          typeid (id),
          &comparator_impl<id>,
          false);

        typedef idref<C, ncname, type> idref;
        register_type (
          typeid (idref),
          &comparator_impl<idref>,
          false);

        typedef idrefs<C, simple_type, idref> idrefs;
        register_type (
          typeid (idrefs),
          &comparator_impl<idrefs>,
          false);


        // URI.
        //
        typedef uri<C, simple_type> uri;
        register_type (
          typeid (uri),
          &comparator_impl<uri>,
          false);


        // Qualified name.
        //
        typedef qname<C, simple_type, uri, ncname> qname;
        register_type (
          typeid (qname),
          &comparator_impl<qname>,
          false);


        // Binary.
        //
        typedef base64_binary<C, simple_type> base64_binary;
        register_type (
          typeid (base64_binary),
          &comparator_impl<base64_binary>,
          false);

        typedef hex_binary<C, simple_type> hex_binary;
        register_type (
          typeid (hex_binary),
          &comparator_impl<hex_binary>,
          false);


        // Date/time.
        //
        typedef gday<C, simple_type> gday;
        register_type (
          typeid (gday),
          &comparator_impl<gday>,
          false);

        typedef gmonth<C, simple_type> gmonth;
        register_type (
          typeid (gmonth),
          &comparator_impl<gmonth>,
          false);

        typedef gyear<C, simple_type> gyear;
        register_type (
          typeid (gyear),
          &comparator_impl<gyear>,
          false);

        typedef gmonth_day<C, simple_type> gmonth_day;
        register_type (
          typeid (gmonth_day),
          &comparator_impl<gmonth_day>,
          false);

        typedef gyear_month<C, simple_type> gyear_month;
        register_type (
          typeid (gyear_month),
          &comparator_impl<gyear_month>,
          false);

        typedef date<C, simple_type> date;
        register_type (
          typeid (date),
          &comparator_impl<date>,
          false);

        typedef time<C, simple_type> time;
        register_type (
          typeid (time),
          &comparator_impl<time>,
          false);

        typedef date_time<C, simple_type> date_time;
        register_type (
          typeid (date_time),
          &comparator_impl<date_time>,
          false);

        typedef duration<C, simple_type> duration;
        register_type (
          typeid (duration),
          &comparator_impl<duration>,
          false);


        // Entity.
        //
        typedef entity<C, ncname> entity;
        register_type (
          typeid (entity),
          &comparator_impl<entity>,
          false);

        typedef entities<C, simple_type, entity> entities;
        register_type (
          typeid (entities),
          &comparator_impl<entities>,
          false);
      }

      template <typename C>
      void comparison_map<C>::
      register_type (const type_id& tid, comparator c, bool override)
      {
        if (override || type_map_.find (&tid) == type_map_.end ())
          type_map_[&tid] = c;
      }

      template <typename C>
      void comparison_map<C>::
      unregister_type (const type_id& tid)
      {
        type_map_.erase (&tid);
      }

      template <typename C>
      bool comparison_map<C>::
      compare (const type& x, const type& y)
      {
        const type_id& xi (typeid (x));

        if (xi != typeid (y))
          return false;

        if (comparator c = find (xi))
          return c (x, y);
        else
          throw no_type_info<C> (std::basic_string<C> (),
                                 std::basic_string<C> ()); // @@ TODO
      }

      template <typename C>
      typename comparison_map<C>::comparator
      comparison_map<C>::
      find (const type_id& tid) const
      {
        typename type_map::const_iterator i (type_map_.find (&tid));
        return i == type_map_.end () ? 0 : i->second;
      }


      // comparison_plate
      //
      template<unsigned long id, typename C>
      comparison_plate<id, C>::
      comparison_plate ()
      {
        if (count == 0)
          map = new comparison_map<C>;

        ++count;
      }

      template<unsigned long id, typename C>
      comparison_plate<id, C>::
      ~comparison_plate ()
      {
        if (--count == 0)
          delete map;
      }

      //
      //
      template<typename T>
      bool
      comparator_impl (const type& x, const type& y)
      {
        return static_cast<const T&> (x) == static_cast<const T&> (y);
      }

      // comparison_initializer
      //
      template<unsigned long id, typename C, typename T>
      comparison_initializer<id, C, T>::
      comparison_initializer ()
      {
        comparison_map_instance<id, C> ().register_type (
          typeid (T), &comparator_impl<T>);
      }

      template<unsigned long id, typename C, typename T>
      comparison_initializer<id, C, T>::
      ~comparison_initializer ()
      {
        comparison_map_instance<id, C> ().unregister_type (typeid (T));
      }
    }
  }
}
