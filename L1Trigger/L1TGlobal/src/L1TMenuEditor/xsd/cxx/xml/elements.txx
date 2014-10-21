// file      : xsd/cxx/xml/elements.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      // properties
      //

      template <typename C>
      void properties<C>::
      schema_location (const std::basic_string<C>& ns,
                       const std::basic_string<C>& loc)
      {
        if (ns.empty () || loc.empty ())
          throw argument ();

        if (!schema_location_.empty ())
          schema_location_ += C (' ');

        schema_location_ += ns + C (' ') + loc;
      }

      template <typename C>
      void properties<C>::
      no_namespace_schema_location (const std::basic_string<C>& loc)
      {
        if (loc.empty ())
          throw argument ();

        if (!no_namespace_schema_location_.empty ())
          no_namespace_schema_location_ += C (' ');

        no_namespace_schema_location_ += loc;
      }


      //
      //

      template <typename C>
      std::basic_string<C>
      prefix (const std::basic_string<C>& n)
      {
        std::size_t i (0);

        while (i < n.length () && n[i] != ':')
          ++i;

        return std::basic_string<C> (n, i == n.length () ? i : 0, i);
      }

      template <typename C>
      std::basic_string<C>
      uq_name (const std::basic_string<C>& n)
      {
        std::size_t i (0);

        while (i < n.length () && n[i] != ':')
          ++i;

        return std::basic_string<C> (
          n.c_str () + (i == n.length () ? 0 : i + 1));
      }
    }
  }
}

