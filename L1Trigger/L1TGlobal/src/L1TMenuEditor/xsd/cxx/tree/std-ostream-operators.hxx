// file      : xsd/cxx/tree/std-ostream-operators.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_STD_OSTREAM_OPERATORS_HXX
#define XSD_CXX_TREE_STD_OSTREAM_OPERATORS_HXX

#include <ostream>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/containers.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/list.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // type
      //
      template <typename C>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const type&)
      {
        return os;
      }


      // simple_type
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const simple_type<B>&)
      {
        return os;
      }


      // fundamental_base
      //
      template <typename T, typename C, typename B, schema_type::value ST>
      inline
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, fundamental_base<T, C, B, ST> x)
      {
        T& r (x);
        return os << r;
      }

      // optional: see containers.hxx
      //

      // list
      //

      // This is an xsd:list-style format (space-separated).
      //
      template <typename C, typename T, schema_type::value ST, bool fund>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const list<T, C, ST, fund>& v)
      {
        for (typename list<T, C, ST, fund>::const_iterator
               b (v.begin ()), e (v.end ()), i (b); i != e; ++i)
        {
          if (i != b)
            os << C (' ');

          os << *i;
        }

        return os;
      }


      // Operators for built-in types.
      //


      // string
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const string<C, B>& v)
      {
        const std::basic_string<C>& r (v);
        return os << r;
      }


      // normalized_string
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const normalized_string<C, B>& v)
      {
        const B& r (v);
        return os << r;
      }


      // token
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const token<C, B>& v)
      {
        const B& r (v);
        return os << r;
      }


      // nmtoken
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const nmtoken<C, B>& v)
      {
        const B& r (v);
        return os << r;
      }


      // nmtokens
      //
      template <typename C, typename B, typename nmtoken>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const nmtokens<C, B, nmtoken>& v)
      {
        const list<nmtoken, C>& r (v);
        return os << r;
      }


      // name
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const name<C, B>& v)
      {
        const B& r (v);
        return os << r;
      }


      // ncname
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const ncname<C, B>& v)
      {
        const B& r (v);
        return os << r;
      }


      // language
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const language<C, B>& v)
      {
        const B& r (v);
        return os << r;
      }


      // id
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const id<C, B>& v)
      {
        const B& r (v);
        return os << r;
      }


      // idref
      //
      template <typename C, typename B, typename T>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const idref<C, B, T>& v)
      {
        const B& r (v);
        return os << r;
      }


      // idrefs
      //
      template <typename C, typename B, typename idref>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const idrefs<C, B, idref>& v)
      {
        const list<idref, C>& r (v);
        return os << r;
      }


      // uri
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const uri<C, B>& v)
      {
        const std::basic_string<C>& r (v);
        return os << r;
      }


      // qname
      //
      template <typename C, typename B, typename uri, typename ncname>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os,
                  const qname<C, B, uri, ncname>& n)
      {
        if (n.qualified ())
          os << n.namespace_ () << C ('#');

        return os << n.name ();
      }


      // base64_binary
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const base64_binary<C, B>& v)
      {
        return os << v.encode ();
      }


      // hex_binary
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const hex_binary<C, B>& v)
      {
        return os << v.encode ();
      }


      // entity
      //
      template <typename C, typename B>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const entity<C, B>& v)
      {
        const B& r (v);
        return os << r;
      }


      // entities
      //
      template <typename C, typename B, typename entity>
      inline std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const entities<C, B, entity>& v)
      {
        const list<entity, C>& r (v);
        return os << r;
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/date-time-ostream.txx>

#endif  // XSD_CXX_TREE_STD_OSTREAM_OPERATORS_HXX
