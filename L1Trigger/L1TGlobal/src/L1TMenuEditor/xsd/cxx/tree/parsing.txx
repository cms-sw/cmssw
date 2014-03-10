// file      : xsd/cxx/tree/parsing.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <string>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.hxx>         // trim

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.hxx>        // xml::{string, transcode}
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/elements.hxx>      // xml::{prefix, uq_name}
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/bits/literals.hxx> // xml::bits::{xml_prefix,
                                         //             xml_namespace}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/exceptions.hxx>   // no_prefix_mapping
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/list.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/text.hxx>         // text_content

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // Note that most of the types implemented here (except string,
      // (normalizedString, and base64Binary) cannot have whitespaces
      // in the value. As result we don't need to waste time collapsing
      // whitespaces. All we need to do is trim the string representation
      // which can be done without copying.
      //

      // type
      //
      inline _type::
      _type (const xercesc::DOMElement& e, flags f, container* c)
          : dom_info_ (0), container_ (c)
      {
        if (f & flags::keep_dom)
        {
          std::auto_ptr<dom_info> r (
            dom_info_factory::create (e, *this, c == 0));
          dom_info_ = r;
        }
      }

      inline _type::
      _type (const xercesc::DOMAttr& a, flags f, container* c)
          : dom_info_ (0), container_ (c)
      {
        if (f & flags::keep_dom)
        {
          std::auto_ptr<dom_info> r (dom_info_factory::create (a, *this));
          dom_info_ = r;
        }
      }

      template <typename C>
      inline _type::
      _type (const std::basic_string<C>&,
             const xercesc::DOMElement*,
             flags,
             container* c)
          : dom_info_ (0), // List elements don't have associated DOM nodes.
            container_ (c)
      {
      }

      // simple_type
      //
      template <typename B>
      inline simple_type<B>::
      simple_type (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
      }

      template <typename B>
      inline simple_type<B>::
      simple_type (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
      }

      template <typename B>
      template <typename C>
      inline simple_type<B>::
      simple_type (const std::basic_string<C>& s,
                   const xercesc::DOMElement* e,
                   flags f,
                   container* c)
          : B (s, e, f, c)
      {
      }

      // fundamental_base
      //
      template <typename T, typename C, typename B, schema_type::value ST>
      fundamental_base<T, C, B, ST>::
      fundamental_base (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c),
            facet_table_ (0),
            x_ (traits<T, C, ST>::create (e, f, c))
      {
      }

      template <typename T, typename C, typename B, schema_type::value ST>
      fundamental_base<T, C, B, ST>::
      fundamental_base (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c),
            facet_table_ (0),
            x_ (traits<T, C, ST>::create (a, f, c))
      {
      }

      template <typename T, typename C, typename B, schema_type::value ST>
      fundamental_base<T, C, B, ST>::
      fundamental_base (const std::basic_string<C>& s,
                        const xercesc::DOMElement* e,
                        flags f,
                        container* c)
          : B (s, e, f, c),
            facet_table_ (0),
            x_ (traits<T, C, ST>::create (s, e, f, c))
      {
      }


      // Parsing c-tors for list.
      //

      namespace bits
      {
        // Find first non-space character.
        //
        template <typename C>
        typename std::basic_string<C>::size_type
        find_ns (const C* s,
                 typename std::basic_string<C>::size_type size,
                 typename std::basic_string<C>::size_type pos)
        {
          while (pos < size &&
                 (s[pos] == C (0x20) || // space
                  s[pos] == C (0x0D) || // carriage return
                  s[pos] == C (0x09) || // tab
                  s[pos] == C (0x0A)))
            ++pos;

          return pos < size ? pos : std::basic_string<C>::npos;
        }

        // Find first space character.
        //
        template <typename C>
        typename std::basic_string<C>::size_type
        find_s (const C* s,
                typename std::basic_string<C>::size_type size,
                typename std::basic_string<C>::size_type pos)
        {
          while (pos < size &&
                 s[pos] != C (0x20) && // space
                 s[pos] != C (0x0D) && // carriage return
                 s[pos] != C (0x09) && // tab
                 s[pos] != C (0x0A))
            ++pos;

          return pos < size ? pos : std::basic_string<C>::npos;
        }
      }

      // Individual items of the list have no DOM association. Therefore
      // I clear keep_dom from flags.
      //

      template <typename T, typename C, schema_type::value ST>
      list<T, C, ST, false>::
      list (const xercesc::DOMElement& e, flags f, container* c)
          : sequence<T> (flags (f & ~flags::keep_dom), c) // ambiguous
      {
        init (text_content<C> (e), &e);
      }

      template <typename T, typename C, schema_type::value ST>
      list<T, C, ST, false>::
      list (const xercesc::DOMAttr& a, flags f, container* c)
          : sequence<T> (flags (f & ~flags::keep_dom), c) // ambiguous
      {
        init (xml::transcode<C> (a.getValue ()), a.getOwnerElement ());
      }

      template <typename T, typename C, schema_type::value ST>
      list<T, C, ST, false>::
      list (const std::basic_string<C>& s,
            const xercesc::DOMElement* e,
            flags f,
            container* c)
          : sequence<T> (flags (f & ~flags::keep_dom), c) // ambiguous
      {
        init (s, e);
      }

      template <typename T, typename C, schema_type::value ST>
      void list<T, C, ST, false>::
      init (const std::basic_string<C>& s, const xercesc::DOMElement* parent)
      {
        if (s.size () == 0)
          return;

        using std::basic_string;
        typedef typename sequence<T>::ptr ptr;
        typedef typename basic_string<C>::size_type size_type;

        const C* data (s.c_str ());
        size_type size (s.size ());

        // Traverse the data while logically collapsing spaces.
        //
        for (size_type i (bits::find_ns<C> (data, size, 0));
             i != basic_string<C>::npos;)
        {
          size_type j (bits::find_s (data, size, i));

          if (j != basic_string<C>::npos)
          {
            ptr r (
              new T (basic_string<C> (data + i, j - i),
                     parent,
                     this->flags_,
                     this->container_));

            this->v_.push_back (r);

            i = bits::find_ns (data, size, j);
          }
          else
          {
            // Last element.
            //
            ptr r (
              new T (basic_string<C> (data + i, size - i),
                     parent,
                     this->flags_,
                     this->container_));

            this->v_.push_back (r);

            break;
          }
        }
      }

      template <typename T, typename C, schema_type::value ST>
      list<T, C, ST, true>::
      list (const xercesc::DOMElement& e, flags f, container* c)
          : sequence<T> (flags (f & ~flags::keep_dom), c) // ambiguous
      {
        init (text_content<C> (e), &e);
      }

      template <typename T, typename C, schema_type::value ST>
      inline list<T, C, ST, true>::
      list (const xercesc::DOMAttr& a, flags f, container* c)
          : sequence<T> (flags (f & ~flags::keep_dom), c) // ambiguous
      {
        init (xml::transcode<C> (a.getValue ()), a.getOwnerElement ());
      }

      template <typename T, typename C, schema_type::value ST>
      inline list<T, C, ST, true>::
      list (const std::basic_string<C>& s,
            const xercesc::DOMElement* parent,
            flags f,
            container* c)
          : sequence<T> (flags (f & ~flags::keep_dom), c) // ambiguous
      {
        init (s, parent);
      }

      template <typename T, typename C, schema_type::value ST>
      inline void list<T, C, ST, true>::
      init (const std::basic_string<C>& s, const xercesc::DOMElement* parent)
      {
        if (s.size () == 0)
          return;

        using std::basic_string;
        typedef typename basic_string<C>::size_type size_type;

        const C* data (s.c_str ());
        size_type size (s.size ());

        // Traverse the data while logically collapsing spaces.
        //
        for (size_type i (bits::find_ns<C> (data, size, 0));
             i != basic_string<C>::npos;)
        {
          size_type j (bits::find_s (data, size, i));

          if (j != basic_string<C>::npos)
          {
            this->push_back (
              traits<T, C, ST>::create (
                basic_string<C> (data + i, j - i), parent, 0, 0));

            i = bits::find_ns (data, size, j);
          }
          else
          {
            // Last element.
            //
            this->push_back (
              traits<T, C, ST>::create (
                basic_string<C> (data + i, size - i), parent, 0, 0));

            break;
          }
        }
      }


      // Parsing c-tors for built-in types.
      //


      // string
      //
      template <typename C, typename B>
      string<C, B>::
      string (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c),
            base_type (text_content<C> (e))
      {
      }

      template <typename C, typename B>
      string<C, B>::
      string (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c),
            base_type (xml::transcode<C> (a.getValue ()))
      {
      }

      template <typename C, typename B>
      string<C, B>::
      string (const std::basic_string<C>& s,
              const xercesc::DOMElement* e,
              flags f,
              container* c)
          : B (s, e, f, c), base_type (s)
      {
      }


      // normalized_string
      //
      template <typename C, typename B>
      normalized_string<C, B>::
      normalized_string (const xercesc::DOMElement& e, flags f, container* c)
          : base_type (e, f, c)
      {
        normalize ();
      }

      template <typename C, typename B>
      normalized_string<C, B>::
      normalized_string (const xercesc::DOMAttr& a, flags f, container* c)
          : base_type (a, f, c)
      {
        normalize ();
      }

      template <typename C, typename B>
      normalized_string<C, B>::
      normalized_string (const std::basic_string<C>& s,
                         const xercesc::DOMElement* e,
                         flags f,
                         container* c)
          : base_type (s, e, f, c)
      {
        normalize ();
      }

      template <typename C, typename B>
      void normalized_string<C, B>::
      normalize ()
      {
        typedef typename std::basic_string<C>::size_type size_type;

        size_type size (this->size ());

        for (size_type i (0); i < size; ++i)
        {
          C& c ((*this)[i]);

          if (c == C (0x0D) || // carriage return
              c == C (0x09) || // tab
              c == C (0x0A))
            c = C (0x20);
        }
      }


      // token
      //
      template <typename C, typename B>
      token<C, B>::
      token (const xercesc::DOMElement& e, flags f, container* c)
          : base_type (e, f, c)
      {
        collapse ();
      }

      template <typename C, typename B>
      token<C, B>::
      token (const xercesc::DOMAttr& a, flags f, container* c)
          : base_type (a, f, c)
      {
        collapse ();
      }

      template <typename C, typename B>
      token<C, B>::
      token (const std::basic_string<C>& s,
             const xercesc::DOMElement* e,
             flags f,
             container* c)
          : base_type (s, e, f, c)
      {
        collapse ();
      }

      template <typename C, typename B>
      void token<C, B>::
      collapse ()
      {
        // We have all whitespace normilized by our base. We just
        // need to collapse them.
        //
        typedef typename std::basic_string<C>::size_type size_type;

        size_type size (this->size ()), j (0);
        bool subs (false), trim (true);

        for (size_type i (0); i < size; ++i)
        {
          C c ((*this)[i]);

          if (c == C (0x20))
          {
            subs = true;
          }
          else
          {
            if (subs)
            {
              subs = false;

              if (!trim)
                (*this)[j++] = C (0x20);
            }

            if (trim)
              trim = false;

            (*this)[j++] = c;
          }
        }

        this->resize (j);
      }


      // nmtoken
      //
      template <typename C, typename B>
      nmtoken<C, B>::
      nmtoken (const xercesc::DOMElement& e, flags f, container* c)
          : base_type (e, f, c)
      {
      }

      template <typename C, typename B>
      nmtoken<C, B>::
      nmtoken (const xercesc::DOMAttr& a, flags f, container* c)
          : base_type (a, f, c)
      {
      }

      template <typename C, typename B>
      nmtoken<C, B>::
      nmtoken (const std::basic_string<C>& s,
               const xercesc::DOMElement* e,
               flags f,
               container* c)
          : base_type (s, e, f, c)
      {
      }


      // nmtokens
      //
      template <typename C, typename B, typename nmtoken>
      nmtokens<C, B, nmtoken>::
      nmtokens (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c), base_type (e, f, this)
      {
      }

      template <typename C, typename B, typename nmtoken>
      nmtokens<C, B, nmtoken>::
      nmtokens (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c), base_type (a, f, this)
      {
      }

      template <typename C, typename B, typename nmtoken>
      nmtokens<C, B, nmtoken>::
      nmtokens (const std::basic_string<C>& s,
                const xercesc::DOMElement* e,
                flags f,
                container* c)
          : B (s, e, f, c), base_type (s, e, f, this)
      {
      }


      // name
      //
      template <typename C, typename B>
      name<C, B>::
      name (const xercesc::DOMElement& e, flags f, container* c)
          : base_type (e, f, c)
      {
      }

      template <typename C, typename B>
      name<C, B>::
      name (const xercesc::DOMAttr& a, flags f, container* c)
          : base_type (a, f, c)
      {
      }

      template <typename C, typename B>
      name<C, B>::
      name (const std::basic_string<C>& s,
            const xercesc::DOMElement* e,
            flags f,
            container* c)
          : base_type (s, e, f, c)
      {
      }


      // ncname
      //
      template <typename C, typename B>
      ncname<C, B>::
      ncname (const xercesc::DOMElement& e, flags f, container* c)
          : base_type (e, f, c)
      {
      }

      template <typename C, typename B>
      ncname<C, B>::
      ncname (const xercesc::DOMAttr& a, flags f, container* c)
          : base_type (a, f, c)
      {
      }

      template <typename C, typename B>
      ncname<C, B>::
      ncname (const std::basic_string<C>& s,
              const xercesc::DOMElement* e,
              flags f,
              container* c)
          : base_type (s, e, f, c)
      {
      }


      // language
      //
      template <typename C, typename B>
      language<C, B>::
      language (const xercesc::DOMElement& e, flags f, container* c)
          : base_type (e, f, c)
      {
      }

      template <typename C, typename B>
      language<C, B>::
      language (const xercesc::DOMAttr& a, flags f, container* c)
          : base_type (a, f, c)
      {
      }

      template <typename C, typename B>
      language<C, B>::
      language (const std::basic_string<C>& s,
                const xercesc::DOMElement* e,
                flags f,
                container* c)
          : base_type (s, e, f, c)
      {
      }


      // id
      //
      template <typename C, typename B>
      id<C, B>::
      id (const xercesc::DOMElement& e, flags f, container* c)
          : base_type (e, f, c), identity_ (*this)
      {
        register_id ();
      }

      template <typename C, typename B>
      id<C, B>::
      id (const xercesc::DOMAttr& a, flags f, container* c)
          : base_type (a, f, c), identity_ (*this)
      {
        register_id ();
      }

      template <typename C, typename B>
      id<C, B>::
      id (const std::basic_string<C>& s,
          const xercesc::DOMElement* e,
          flags f,
          container* c)
          : base_type (s, e, f, c), identity_ (*this)
      {
        register_id ();
      }


      // idref
      //
      template <typename C, typename B, typename T>
      idref<C, B, T>::
      idref (const xercesc::DOMElement& e, flags f, container* c)
          : base_type (e, f, c), identity_ (*this)
      {
      }

      template <typename C, typename B, typename T>
      idref<C, B, T>::
      idref (const xercesc::DOMAttr& a, flags f, container* c)
          : base_type (a, f , c), identity_ (*this)
      {
      }

      template <typename C, typename B, typename T>
      idref<C, B, T>::
      idref (const std::basic_string<C>& s,
             const xercesc::DOMElement* e,
             flags f,
             container* c)
          : base_type (s, e, f, c), identity_ (*this)
      {
      }


      // idrefs
      //
      template <typename C, typename B, typename idref>
      idrefs<C, B, idref>::
      idrefs (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c), base_type (e, f, this)
      {
      }

      template <typename C, typename B, typename idref>
      idrefs<C, B, idref>::
      idrefs (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c), base_type (a, f, this)
      {
      }

      template <typename C, typename B, typename idref>
      idrefs<C, B, idref>::
      idrefs (const std::basic_string<C>& s,
              const xercesc::DOMElement* e,
              flags f,
              container* c)
          : B (s, e, f, c), base_type (s, e, f, this)
      {
      }


      // uri
      //
      template <typename C, typename B>
      uri<C, B>::
      uri (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c),
            base_type (trim (text_content<C> (e)))
      {
      }

      template <typename C, typename B>
      uri<C, B>::
      uri (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c),
            base_type (trim (xml::transcode<C> (a.getValue ())))
      {
      }

      template <typename C, typename B>
      uri<C, B>::
      uri (const std::basic_string<C>& s,
           const xercesc::DOMElement* e,
           flags f,
           container* c)
          : B (s, e, f, c), base_type (trim (s))
      {
      }


      // qname
      //
      template <typename C, typename B, typename uri, typename ncname>
      qname<C, B, uri, ncname>::
      qname (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        std::basic_string<C> v (trim (text_content<C> (e)));
        ns_ = resolve (v, &e);
        name_ = xml::uq_name (v);
      }

      template <typename C, typename B, typename uri, typename ncname>
      qname<C, B, uri, ncname>::
      qname (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        std::basic_string<C> v (trim (xml::transcode<C> (a.getValue ())));
        ns_ = resolve (v, a.getOwnerElement ());
        name_ = xml::uq_name (v);
      }

      template <typename C, typename B, typename uri, typename ncname>
      qname<C, B, uri, ncname>::
      qname (const std::basic_string<C>& s,
             const xercesc::DOMElement* e,
             flags f,
             container* c)
          : B (s, e, f, c)
      {
        std::basic_string<C> v (trim (s));
        ns_ = resolve (v, e);
        name_ = xml::uq_name (v);
      }

      template <typename C, typename B, typename uri, typename ncname>
      uri qname<C, B, uri, ncname>::
      resolve (const std::basic_string<C>& s, const xercesc::DOMElement* e)
      {
        std::basic_string<C> p (xml::prefix (s));

        if (e)
        {
          // This code is copied verbatim from xml/dom/elements.hxx.
          //

          // 'xml' prefix requires special handling and Xerces folks refuse
          // to handle this in DOM so I have to do it myself.
          //
          if (p == xml::bits::xml_prefix<C> ())
            return xml::bits::xml_namespace<C> ();

          const XMLCh* xns (
            e->lookupNamespaceURI (
              p.empty () ? 0 : xml::string (p).c_str ()));

          if (xns != 0)
            return xml::transcode<C> (xns);
          else if (p.empty ())
            return std::basic_string<C> ();
        }

        throw no_prefix_mapping<C> (p);
      }


      // base64_binary
      //
      // We are not doing whitespace collapsing since the decode
      // functions can handle it like this.
      //
      template <typename C, typename B>
      base64_binary<C, B>::
      base64_binary (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        // This implementation is not optimal.
        //
        std::basic_string<C> str (trim (text_content<C> (e)));
        decode (xml::string (str).c_str ());
      }

      template <typename C, typename B>
      base64_binary<C, B>::
      base64_binary (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        std::basic_string<C> str (trim (xml::transcode<C> (a.getValue ())));
        decode (xml::string (str).c_str ());
      }

      template <typename C, typename B>
      base64_binary<C, B>::
      base64_binary (const std::basic_string<C>& s,
                     const xercesc::DOMElement* e,
                     flags f,
                     container* c)
          : B (s, e, f, c)
      {
        std::basic_string<C> str (trim (s));
        decode (xml::string (str).c_str ());
      }


      // hex_binary
      //
      template <typename C, typename B>
      hex_binary<C, B>::
      hex_binary (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        // This implementation is not optimal.
        //
        std::basic_string<C> str (trim (text_content<C> (e)));
        decode (xml::string (str).c_str ());
      }

      template <typename C, typename B>
      hex_binary<C, B>::
      hex_binary (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        std::basic_string<C> str (trim (xml::transcode<C> (a.getValue ())));
        decode (xml::string (str).c_str ());
      }

      template <typename C, typename B>
      hex_binary<C, B>::
      hex_binary (const std::basic_string<C>& s,
                  const xercesc::DOMElement* e,
                  flags f,
                  container* c)
          : B (s, e, f, c)
      {
        std::basic_string<C> str (trim (s));
        decode (xml::string (str).c_str ());
      }

      // entity
      //
      template <typename C, typename B>
      entity<C, B>::
      entity (const xercesc::DOMElement& e, flags f, container* c)
          : base_type (e, f, c)
      {
      }

      template <typename C, typename B>
      entity<C, B>::
      entity (const xercesc::DOMAttr& a, flags f, container* c)
          : base_type (a, f, c)
      {
      }

      template <typename C, typename B>
      entity<C, B>::
      entity (const std::basic_string<C>& s,
              const xercesc::DOMElement* e,
              flags f,
              container* c)
          : base_type (s, e, f, c)
      {
      }


      // entities
      //
      template <typename C, typename B, typename entity>
      entities<C, B, entity>::
      entities (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c), base_type (e, f, this)
      {
      }

      template <typename C, typename B, typename entity>
      entities<C, B, entity>::
      entities (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c), base_type (a, f, this)
      {
      }

      template <typename C, typename B, typename entity>
      entities<C, B, entity>::
      entities (const std::basic_string<C>& s,
                const xercesc::DOMElement* e,
                flags f,
                container* c)
          : B (s, e, f, c), base_type (s, e, f, this)
      {
      }
    }
  }
}
