// file      : xsd/cxx/tree/stream-insertion.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_STREAM_INSERTION_HXX
#define XSD_CXX_TREE_STREAM_INSERTION_HXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/list.hxx>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/ostream.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // type
      //
      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, const type&)
      {
        return s;
      }

      // simple_type
      //
      template <typename S, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const simple_type<B>&)
      {
        return s;
      }

      // fundamental_base
      //
      template <typename S,
                typename T,
                typename C,
                typename B,
                schema_type::value ST>
      inline ostream<S>&
      operator<< (ostream<S>& s, const fundamental_base<T, C, B, ST>& x)
      {
        const T& r (x);
        return s << r;
      }

      // list
      //
      template <typename S,
                typename T,
                typename C,
                schema_type::value ST,
                bool fund>
      ostream<S>&
      operator<< (ostream<S>& s, const list<T, C, ST, fund>& x)
      {
        s << ostream_common::as_size<std::size_t> (x.size ());

        for (typename list<T, C, ST, fund>::const_iterator
               i (x.begin ()), e (x.end ()); i != e; ++i)
        {
          s << *i;
        }

        return s;
      }


      // Insertion operators for built-in types.
      //


      // string
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const string<C, B>& x)
      {
        const std::basic_string<C>& r (x);
        return s << r;
      }


      // normalized_string
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const normalized_string<C, B>& x)
      {
        const B& r (x);
        return s << r;
      }


      // token
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const token<C, B>& x)
      {
        const B& r (x);
        return s << r;
      }


      // nmtoken
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const nmtoken<C, B>& x)
      {
        const B& r (x);
        return s << r;
      }


      // nmtokens
      //
      template <typename S, typename C, typename B, typename nmtoken>
      inline ostream<S>&
      operator<< (ostream<S>& s, const nmtokens<C, B, nmtoken>& x)
      {
        const list<nmtoken, C>& r (x);
        return s << r;
      }


      // name
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const name<C, B>& x)
      {
        const B& r (x);
        return s << r;
      }


      // ncname
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const ncname<C, B>& x)
      {
        const B& r (x);
        return s << r;
      }


      // language
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const language<C, B>& x)
      {
        const std::basic_string<C>& r (x);
        return s << r;
      }


      // id
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const id<C, B>& x)
      {
        const std::basic_string<C>& r (x);
        return s << r;
      }


      // idref
      //
      template <typename S, typename C, typename B, typename T>
      inline ostream<S>&
      operator<< (ostream<S>& s, const idref<C, B, T>& x)
      {
        const B& r (x);
        return s << r;
      }


      // idrefs
      //
      template <typename S, typename C, typename B, typename idref>
      inline ostream<S>&
      operator<< (ostream<S>& s, const idrefs<C, B, idref>& x)
      {
        const list<idref, C>& r (x);
        return s << r;
      }


      // uri
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const uri<C, B>& x)
      {
        const std::basic_string<C>& r (x);
        return s << r;
      }


      // qname
      //
      template <typename S,
                typename C,
                typename B,
                typename uri,
                typename ncname>
      inline ostream<S>&
      operator<< (ostream<S>& s, const qname<C, B, uri, ncname>& x)
      {
        return s << x.namespace_ () << x.name ();
      }


      // base64_binary
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const base64_binary<C, B>& x)
      {
        const buffer<C>& r (x);
        return s << r;
      }


      // hex_binary
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const hex_binary<C, B>& x)
      {
        const buffer<C>& r (x);
        return s << r;
      }


      // entity
      //
      template <typename S, typename C, typename B>
      inline ostream<S>&
      operator<< (ostream<S>& s, const entity<C, B>& x)
      {
        const B& r (x);
        return s << r;
      }


      // entities
      //
      template <typename S, typename C, typename B, typename entity>
      inline ostream<S>&
      operator<< (ostream<S>& s, const entities<C, B, entity>& x)
      {
        const list<entity, C>& r (x);
        return s << r;
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/date-time-insertion.txx>

#endif  // XSD_CXX_TREE_STREAM_INSERTION_HXX
