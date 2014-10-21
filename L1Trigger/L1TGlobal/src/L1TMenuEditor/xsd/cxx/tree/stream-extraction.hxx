// file      : xsd/cxx/tree/stream-extraction.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_STREAM_EXTRACTION_HXX
#define XSD_CXX_TREE_STREAM_EXTRACTION_HXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/list.hxx>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/istream.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // type
      //
      template <typename S>
      inline _type::
      _type (istream<S>&, flags, container* c)
          : container_ (c)
      {
      }

      // simple_type
      //
      template <typename B>
      template <typename S>
      inline simple_type<B>::
      simple_type (istream<S>& s, flags f, container* c)
          : type (s, f, c)
      {
      }

      // fundamental_base
      //
      template <typename T, typename C, typename B, schema_type::value ST>
      template <typename S>
      inline fundamental_base<T, C, B, ST>::
      fundamental_base (istream<S>& s, flags f, container* c)
          : B (s, f, c), facet_table_ (0)
      {
        T& r (*this);
        s >> r;
      }

      // list
      //
      template <typename T, typename C, schema_type::value ST>
      template <typename S>
      list<T, C, ST, false>::
      list (istream<S>& s, flags f, container* c)
          : sequence<T> (f, c)
      {
        std::size_t size;
        istream_common::as_size<std::size_t> as_size (size);
        s >> as_size;

        if (size > 0)
        {
          this->reserve (size);

          while (size--)
          {
            std::auto_ptr<T> p (new T (s, f, c));
            this->push_back (p);
          }
        }
      }

      template <typename T, typename C, schema_type::value ST>
      template <typename S>
      list<T, C, ST, true>::
      list (istream<S>& s, flags f, container* c)
          : sequence<T> (f, c)
      {
        std::size_t size;
        istream_common::as_size<std::size_t> as_size (size);
        s >> as_size;

        if (size > 0)
        {
          this->reserve (size);

          while (size--)
          {
            T x;
            s >> x;
            this->push_back (x);
          }
        }
      }

      // Extraction operators for built-in types.
      //


      // string
      //
      template <typename C, typename B>
      template <typename S>
      inline string<C, B>::
      string (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        std::basic_string<C>& r (*this);
        s >> r;
      }


      // normalized_string
      //
      template <typename C, typename B>
      template <typename S>
      inline normalized_string<C, B>::
      normalized_string (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
      }


      // token
      //
      template <typename C, typename B>
      template <typename S>
      inline token<C, B>::
      token (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
      }


      // nmtoken
      //
      template <typename C, typename B>
      template <typename S>
      inline nmtoken<C, B>::
      nmtoken (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
      }


      // nmtokens
      //
      template <typename C, typename B, typename nmtoken>
      template <typename S>
      inline nmtokens<C, B, nmtoken>::
      nmtokens (istream<S>& s, flags f, container* c)
          : B (s, f, c), base_type (s, f, this)
      {
      }


      // name
      //
      template <typename C, typename B>
      template <typename S>
      inline name<C, B>::
      name (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
      }


      // ncname
      //
      template <typename C, typename B>
      template <typename S>
      inline ncname<C, B>::
      ncname (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
      }


      // language
      //
      template <typename C, typename B>
      template <typename S>
      inline language<C, B>::
      language (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
      }


      // id
      //
      template <typename C, typename B>
      template <typename S>
      inline id<C, B>::
      id (istream<S>& s, flags f, container* c)
          : B (s, f, c), identity_ (*this)
      {
        register_id ();
      }


      // idref
      //
      template <typename C, typename B, typename T>
      template <typename S>
      inline idref<C, B, T>::
      idref (istream<S>& s, flags f, container* c)
          : B (s, f, c), identity_ (*this)
      {
      }


      // idrefs
      //
      template <typename C, typename B, typename idref>
      template <typename S>
      inline idrefs<C, B, idref>::
      idrefs (istream<S>& s, flags f, container* c)
          : B (s, f, c), base_type (s, f, this)
      {
      }


      // uri
      //
      template <typename C, typename B>
      template <typename S>
      inline uri<C, B>::
      uri (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        std::basic_string<C>& r (*this);
        s >> r;
      }


      // qname
      //
      template <typename C, typename B, typename uri, typename ncname>
      template <typename S>
      inline qname<C, B, uri, ncname>::
      qname (istream<S>& s, flags f, container* c)
          : B (s, f, c), ns_ (s), name_ (s)
      {
      }


      // base64_binary
      //
      template <typename C, typename B>
      template <typename S>
      inline base64_binary<C, B>::
      base64_binary (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        buffer<C>& r (*this);
        s >> r;
      }


      // hex_binary
      //
      template <typename C, typename B>
      template <typename S>
      inline hex_binary<C, B>::
      hex_binary (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        buffer<C>& r (*this);
        s >> r;
      }


      // entity
      //
      template <typename C, typename B>
      template <typename S>
      inline entity<C, B>::
      entity (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
      }


      // entities
      //
      template <typename C, typename B, typename entity>
      template <typename S>
      inline entities<C, B, entity>::
      entities (istream<S>& s, flags f, container* c)
          : B (s, f, c), base_type (s, f, this)
      {
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/date-time-extraction.txx>

#endif  // XSD_CXX_TREE_STREAM_EXTRACTION_HXX
