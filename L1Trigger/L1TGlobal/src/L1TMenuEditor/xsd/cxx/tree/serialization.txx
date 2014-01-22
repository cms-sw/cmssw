// file      : xsd/cxx/tree/serialization.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <string>
#include <sstream>

#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMElement.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.hxx>        // xml::{string, transcode}
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/serialization-header.hxx>  // dom::{prefix, clear}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/list.hxx>

// The only way to make the following serialization operators
// for fundamental types work is to defined them in the xercesc
// namespace so that they can be found by ADL. Placing them into
// the global namespace does not work.
//

namespace XERCES_CPP_NAMESPACE
{
  // Serialization of std::basic_string and C string. Used in other
  // serializers. Also used to serialize enumerators.
  //
  template <typename C>
  void
  operator<< (xercesc::DOMElement& e, const C* s)
  {
    xsd::cxx::xml::dom::clear<char> (e);

    if (*s != C (0))
      e.setTextContent (xsd::cxx::xml::string (s).c_str ());
  }

  template <typename C>
  void
  operator<< (xercesc::DOMAttr& a, const C* s)
  {
    a.setValue (xsd::cxx::xml::string (s).c_str ());
  }

  // We duplicate the code above instead of delegating in order to
  // allow the xml::string type to take advantage of cached string
  // sizes.
  //
  template <typename C>
  void
  operator<< (xercesc::DOMElement& e, const std::basic_string<C>& s)
  {
    xsd::cxx::xml::dom::clear<char> (e);

    if (!s.empty ())
      e.setTextContent (xsd::cxx::xml::string (s).c_str ());
  }

  template <typename C>
  void
  operator<< (xercesc::DOMAttr& a, const std::basic_string<C>& s)
  {
    a.setValue (xsd::cxx::xml::string (s).c_str ());
  }
}

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // List serialization operators for std::basic_string and C string.
      //

      template <typename C>
      void
      operator<< (list_stream<C>& ls, const C* s)
      {
        ls.os_ << s;
      }

      template <typename C>
      void
      operator<< (list_stream<C>& ls, const std::basic_string<C>& s)
      {
        ls.os_ << s;
      }

      // Insertion operators for type.
      //
      inline void
      operator<< (xercesc::DOMElement& e, const type&)
      {
        xml::dom::clear<char> (e);
      }

      inline void
      operator<< (xercesc::DOMAttr&, const type&)
      {
      }

      template <typename C>
      inline void
      operator<< (list_stream<C>&, const type&)
      {
      }

      // Insertion operators for simple_type.
      //
      template <typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const simple_type<B>&)
      {
        xml::dom::clear<char> (e);
      }

      template <typename B>
      inline void
      operator<< (xercesc::DOMAttr&, const simple_type<B>&)
      {
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>&, const simple_type<B>&)
      {
      }

      // Insertion operators for list.
      //
      template <typename C, typename T, schema_type::value ST, bool fund>
      void
      operator<< (xercesc::DOMElement& e, const list<T, C, ST, fund>& v)
      {
        std::basic_ostringstream<C> os;
        list_stream<C> ls (os, e);

        ls << v;

        e << os.str ();
      }

      template <typename C, typename T, schema_type::value ST, bool fund>
      void
      operator<< (xercesc::DOMAttr& a, const list<T, C, ST, fund>& v)
      {
        std::basic_ostringstream<C> os;
        list_stream<C> ls (os, *a.getOwnerElement ());

        ls << v;

        a << os.str ();
      }

      template <typename C, typename T, schema_type::value ST, bool fund>
      void
      operator<< (list_stream<C>& ls, const list<T, C, ST, fund>& v)
      {
        for (typename list<T, C, ST, fund>::const_iterator
               b (v.begin ()), e (v.end ()), i (b); i != e; ++i)
        {
          if (i != b)
            ls.os_ << C (' ');

          ls << *i;
        }
      }

      // Specializations for double and decimal.
      //
      template <typename C, typename T, bool fund>
      void
      operator<< (list_stream<C>& ls,
                  const list<T, C, schema_type::double_, fund>& v)
      {
        for (typename list<T, C, schema_type::double_, fund>::const_iterator
               b (v.begin ()), e (v.end ()), i (b); i != e; ++i)
        {
          if (i != b)
            ls.os_ << C (' ');

          ls << as_double<T> (*i);
        }
      }

      template <typename C, typename T, bool fund>
      void
      operator<< (list_stream<C>& ls,
                  const list<T, C, schema_type::decimal, fund>& v)
      {
        for (typename list<T, C, schema_type::decimal, fund>::const_iterator
               b (v.begin ()), e (v.end ()), i (b); i != e; ++i)
        {
          if (i != b)
            ls.os_ << C (' ');

          ls << as_decimal<T> (*i);
        }
      }


      // Insertion operators for fundamental_base.
      //
      template <typename T, typename C, typename B, schema_type::value ST>
      void
      operator<< (xercesc::DOMElement& e,
                  const fundamental_base<T, C, B, ST>& x)
      {
        const T& r (x);
        e << r;
      }

      template <typename T, typename C, typename B, schema_type::value ST>
      void
      operator<< (xercesc::DOMAttr& a, const fundamental_base<T, C, B, ST>& x)
      {
        const T& r (x);
        a << r;
      }

      template <typename T, typename C, typename B, schema_type::value ST>
      void
      operator<< (list_stream<C>& ls, const fundamental_base<T, C, B, ST>& x)
      {
        const T& r (x);
        ls << r;
      }

      // Specializations for double.
      //
      template <typename T, typename C, typename B>
      void
      operator<< (
        xercesc::DOMElement& e,
        const fundamental_base<T, C, B, schema_type::double_>& x)
      {
        e << as_double<T> (x);
      }

      template <typename T, typename C, typename B>
      void
      operator<< (
        xercesc::DOMAttr& a,
        const fundamental_base<T, C, B, schema_type::double_>& x)
      {
        a << as_double<T> (x);
      }

      template <typename T, typename C, typename B>
      void
      operator<< (
        list_stream<C>& ls,
        const fundamental_base<T, C, B, schema_type::double_>& x)
      {
        ls << as_double<T> (x);
      }

      // Specializations for decimal.
      //
      template <typename T, typename C, typename B>
      void
      operator<< (
        xercesc::DOMElement& e,
        const fundamental_base<T, C, B, schema_type::decimal>& x)
      {
        e << as_decimal<T> (x, x._facet_table ());
      }

      template <typename T, typename C, typename B>
      void
      operator<< (
        xercesc::DOMAttr& a,
        const fundamental_base<T, C, B, schema_type::decimal>& x)
      {
        a << as_decimal<T> (x, x._facet_table ());
      }

      template <typename T, typename C, typename B>
      void
      operator<< (
        list_stream<C>& ls,
        const fundamental_base<T, C, B, schema_type::decimal>& x)
      {
        ls << as_decimal<T> (x, x._facet_table ());
      }

      // Insertion operators for built-in types.
      //

      namespace bits
      {
        template <typename C, typename T>
        void
        insert (xercesc::DOMElement& e, const T& x)
        {
          std::basic_ostringstream<C> os;
          os << x;
          e << os.str ();
        }

        template <typename C, typename T>
        void
        insert (xercesc::DOMAttr& a, const T& x)
        {
          std::basic_ostringstream<C> os;
          os << x;
          a << os.str ();
        }
      }


      // string
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const string<C, B>& x)
      {
        bits::insert<C> (e, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const string<C, B>& x)
      {
        bits::insert<C> (a, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const string<C, B>& x)
      {
        ls.os_ << x;
      }


      // normalized_string
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const normalized_string<C, B>& x)
      {
        bits::insert<C> (e, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const normalized_string<C, B>& x)
      {
        bits::insert<C> (a, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const normalized_string<C, B>& x)
      {
        ls.os_ << x;
      }


      // token
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const token<C, B>& x)
      {
        bits::insert<C> (e, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const token<C, B>& x)
      {
        bits::insert<C> (a, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const token<C, B>& x)
      {
        ls.os_ << x;
      }


      // nmtoken
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const nmtoken<C, B>& x)
      {
        bits::insert<C> (e, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const nmtoken<C, B>& x)
      {
        bits::insert<C> (a, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const nmtoken<C, B>& x)
      {
        ls.os_ << x;
      }


      // nmtokens
      //
      template <typename C, typename B, typename nmtoken>
      inline void
      operator<< (xercesc::DOMElement& e, const nmtokens<C, B, nmtoken>& v)
      {
        const list<nmtoken, C>& r (v);
        e << r;
      }

      template <typename C, typename B, typename nmtoken>
      inline void
      operator<< (xercesc::DOMAttr& a, const nmtokens<C, B, nmtoken>& v)
      {
        const list<nmtoken, C>& r (v);
        a << r;
      }

      template <typename C, typename B, typename nmtoken>
      inline void
      operator<< (list_stream<C>& ls, const nmtokens<C, B, nmtoken>& v)
      {
        const list<nmtoken, C>& r (v);
        ls << r;
      }


      // name
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const name<C, B>& x)
      {
        bits::insert<C> (e, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const name<C, B>& x)
      {
        bits::insert<C> (a, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const name<C, B>& x)
      {
        ls.os_ << x;
      }


      // ncname
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const ncname<C, B>& x)
      {
        bits::insert<C> (e, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const ncname<C, B>& x)
      {
        bits::insert<C> (a, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const ncname<C, B>& x)
      {
        ls.os_ << x;
      }


      // language
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const language<C, B>& x)
      {
        bits::insert<C> (e, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const language<C, B>& x)
      {
        bits::insert<C> (a, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const language<C, B>& x)
      {
        ls.os_ << x;
      }


      // id
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const id<C, B>& x)
      {
        bits::insert<C> (e, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const id<C, B>& x)
      {
        bits::insert<C> (a, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const id<C, B>& x)
      {
        ls.os_ << x;
      }


      // idref
      //
      template <typename C, typename B, typename T>
      inline void
      operator<< (xercesc::DOMElement& e, const idref<C, B, T>& x)
      {
        bits::insert<C> (e, x);
      }

      template <typename C, typename B, typename T>
      inline void
      operator<< (xercesc::DOMAttr& a, const idref<C, B, T>& x)
      {
        bits::insert<C> (a, x);
      }

      template <typename C, typename B, typename T>
      inline void
      operator<< (list_stream<C>& ls, const idref<C, B, T>& x)
      {
        ls.os_ << x;
      }


      // idrefs
      //
      template <typename C, typename B, typename idref>
      inline void
      operator<< (xercesc::DOMElement& e, const idrefs<C, B, idref>& v)
      {
        const list<idref, C>& r (v);
        e << r;
      }

      template <typename C, typename B, typename idref>
      inline void
      operator<< (xercesc::DOMAttr& a, const idrefs<C, B, idref>& v)
      {
        const list<idref, C>& r (v);
        a << r;
      }

      template <typename C, typename B, typename idref>
      inline void
      operator<< (list_stream<C>& ls, const idrefs<C, B, idref>& v)
      {
        const list<idref, C>& r (v);
        ls << r;
      }


      // uri
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const uri<C, B>& x)
      {
        bits::insert<C> (e, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const uri<C, B>& x)
      {
        bits::insert<C> (a, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const uri<C, B>& x)
      {
        ls.os_ << x;
      }


      // qname
      //
      template <typename C, typename B, typename uri, typename ncname>
      void
      operator<< (xercesc::DOMElement& e, const qname<C, B, uri, ncname>& x)
      {
        std::basic_ostringstream<C> os;

        if (x.qualified ())
        {
          std::basic_string<C> p (xml::dom::prefix (x.namespace_ (), e));

          if (!p.empty ())
            os << p << C (':');
        }

        os << x.name ();
        e << os.str ();
      }

      template <typename C, typename B, typename uri, typename ncname>
      void
      operator<< (xercesc::DOMAttr& a, const qname<C, B, uri, ncname>& x)
      {
        std::basic_ostringstream<C> os;

        if (x.qualified ())
        {
          std::basic_string<C> p (
            xml::dom::prefix (x.namespace_ (), *a.getOwnerElement ()));

          if (!p.empty ())
            os << p << C (':');
        }

        os << x.name ();
        a << os.str ();
      }

      template <typename C, typename B, typename uri, typename ncname>
      void
      operator<< (list_stream<C>& ls, const qname<C, B, uri, ncname>& x)
      {
        if (x.qualified ())
        {
          std::basic_string<C> p (
            xml::dom::prefix (x.namespace_ (), ls.parent_));

          if (!p.empty ())
            ls.os_ << p << C (':');
        }

        ls.os_ << x.name ();
      }


      // base64_binary
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const base64_binary<C, B>& x)
      {
        e << x.encode ();
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const base64_binary<C, B>& x)
      {
        a << x.encode ();
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const base64_binary<C, B>& x)
      {
        ls.os_ << x.encode ();
      }


      // hex_binary
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const hex_binary<C, B>& x)
      {
        e << x.encode ();
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const hex_binary<C, B>& x)
      {
        a << x.encode ();
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const hex_binary<C, B>& x)
      {
        ls.os_ << x.encode ();
      }


      // entity
      //
      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const entity<C, B>& x)
      {
        bits::insert<C> (e, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const entity<C, B>& x)
      {
        bits::insert<C> (a, x);
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const entity<C, B>& x)
      {
        ls.os_ << x;
      }


      // entities
      //
      template <typename C, typename B, typename entity>
      inline void
      operator<< (xercesc::DOMElement& e, const entities<C, B, entity>& v)
      {
        const list<entity, C>& r (v);
        e << r;
      }

      template <typename C, typename B, typename entity>
      inline void
      operator<< (xercesc::DOMAttr& a, const entities<C, B, entity>& v)
      {
        const list<entity, C>& r (v);
        a << r;
      }

      template <typename C, typename B, typename entity>
      inline void
      operator<< (list_stream<C>& ls, const entities<C, B, entity>& v)
      {
        const list<entity, C>& r (v);
        ls << r;
      }
    }
  }
}
