// file      : xsd/cxx/parser/xml-schema.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // string_sequence
      //
      template <typename C>
      bool
      operator== (const string_sequence<C>& a, const string_sequence<C>& b)
      {
        if (a.size () != b.size ())
          return false;

        for (typename string_sequence<C>::const_iterator
               ai (a.begin ()), bi (b.begin ()), ae (a.end ());
             ai != ae; ++ai, ++bi)
        {
          if (*ai != *bi)
            return false;
        }

        return true;
      }
    }
  }
}

