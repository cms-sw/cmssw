// file      : xsd/cxx/parser/xerces/elements.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <istream>
#include <cstddef> // std::size_t
#include <cassert>

#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/validators/schema/SchemaSymbols.hpp>
#include <xercesc/util/XMLUni.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/sax/std-input-source.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/sax/bits/error-handler-proxy.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/bits/literals.hxx> // xml::bits::{xml_prefix, etc}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/error-handler.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/schema-exceptions.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace xerces
      {

        // document
        //

        template <typename C>
        document<C>::
        document (parser_base<C>& parser,
                  const C* name,
                  bool polymorphic)
            : cxx::parser::document<C> (parser, std::basic_string<C> (), name),
              polymorphic_ (polymorphic)
        {
        }

        template <typename C>
        document<C>::
        document (parser_base<C>& parser,
                  const std::basic_string<C>& name,
                  bool polymorphic)
            : cxx::parser::document<C> (parser, std::basic_string<C> (), name),
              polymorphic_ (polymorphic)
        {
        }

        template <typename C>
        document<C>::
        document (parser_base<C>& parser,
                  const C* ns,
                  const C* name,
                  bool polymorphic)
            : cxx::parser::document<C> (parser, ns, name),
              polymorphic_ (polymorphic)
        {
        }

        template <typename C>
        document<C>::
        document (parser_base<C>& parser,
                  const std::basic_string<C>& ns,
                  const std::basic_string<C>& name,
                  bool polymorphic)
            : cxx::parser::document<C> (parser, ns, name),
              polymorphic_ (polymorphic)
        {
        }

        template <typename C>
        document<C>::
        document (bool polymorphic)
            : polymorphic_ (polymorphic)
        {
        }

        // parse (uri)
        //
        template <typename C>
        void document<C>::
        parse (const std::basic_string<C>& uri,
               flags f,
               const properties<C>& p)
        {
          xml::auto_initializer init ((f & flags::dont_initialize) == 0);

          error_handler<C> eh;
          xml::sax::bits::error_handler_proxy<C> eh_proxy (eh);
          std::auto_ptr<xercesc::SAX2XMLReader> sax (create_sax_ (f, p));

          parse (uri, eh_proxy, *sax, f, p);

          eh.throw_if_failed ();
        }

        template <typename C>
        void document<C>::
        parse (const C* uri,
               flags f,
               const properties<C>& p)
        {
          parse (std::basic_string<C> (uri), f, p);
        }

        // error_handler
        //

        template <typename C>
        void document<C>::
        parse (const std::basic_string<C>& uri,
               xml::error_handler<C>& eh,
               flags f,
               const properties<C>& p)
        {
          xml::auto_initializer init ((f & flags::dont_initialize) == 0);

          xml::sax::bits::error_handler_proxy<C> eh_proxy (eh);
          std::auto_ptr<xercesc::SAX2XMLReader> sax (create_sax_ (f, p));

          parse (uri, eh_proxy, *sax, f, p);

          if (eh_proxy.failed ())
            throw parsing<C> ();
        }

        template <typename C>
        void document<C>::
        parse (const C* uri,
               xml::error_handler<C>& eh,
               flags f,
               const properties<C>& p)
        {
          parse (std::basic_string<C> (uri), eh, f, p);
        }

        // ErrorHandler
        //

        template <typename C>
        void document<C>::
        parse (const std::basic_string<C>& uri,
               xercesc::ErrorHandler& eh,
               flags f,
               const properties<C>& p)
        {
          xml::sax::bits::error_handler_proxy<C> eh_proxy (eh);
          std::auto_ptr<xercesc::SAX2XMLReader> sax (create_sax_ (f, p));

          parse (uri, eh_proxy, *sax, f, p);

          if (eh_proxy.failed ())
            throw parsing<C> ();
        }

        template <typename C>
        void document<C>::
        parse (const C* uri,
               xercesc::ErrorHandler& eh,
               flags f,
               const properties<C>& p)
        {
          parse (std::basic_string<C> (uri), eh, f, p);
        }

        // SAX2XMLReader
        //

        template <typename C>
        void document<C>::
        parse (const std::basic_string<C>& uri,
               xercesc::SAX2XMLReader& sax,
               flags f,
               const properties<C>& p)
        {
          // If there is no error handler, then fall back on the default
          // implementation.
          //
          xercesc::ErrorHandler* eh (sax.getErrorHandler ());

          if (eh)
          {
            xml::sax::bits::error_handler_proxy<C> eh_proxy (*eh);

            parse (uri, eh_proxy, sax, f, p);

            if (eh_proxy.failed ())
              throw parsing<C> ();
          }
          else
          {
            error_handler<C> fallback_eh;
            xml::sax::bits::error_handler_proxy<C> eh_proxy (fallback_eh);

            parse (uri, eh_proxy, sax, f, p);

            fallback_eh.throw_if_failed ();
          }
        }

        template <typename C>
        void document<C>::
        parse (const C* uri,
               xercesc::SAX2XMLReader& sax,
               flags f,
               const properties<C>& p)
        {
          parse (std::basic_string<C> (uri), sax, f, p);
        }

        // parse (istream)
        //

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               flags f,
               const properties<C>& p)
        {
          xml::auto_initializer init ((f & flags::dont_initialize) == 0);

          xml::sax::std_input_source isrc (is);

          parse (isrc, f, p);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               xml::error_handler<C>& eh,
               flags f,
               const properties<C>& p)
        {
          xml::auto_initializer init ((f & flags::dont_initialize) == 0);
          xml::sax::std_input_source isrc (is);
          parse (isrc, eh, f, p);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               xercesc::ErrorHandler& eh,
               flags f,
               const properties<C>& p)
        {
          xml::sax::std_input_source isrc (is);
          parse (isrc, eh, f, p);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               xercesc::SAX2XMLReader& sax,
               flags f,
               const properties<C>& p)
        {
          xml::sax::std_input_source isrc (is);
          parse (isrc, sax, f, p);
        }


        // parse (istream, system_id)
        //


        template <typename C>
        void document<C>::
        parse (std::istream& is,
               const std::basic_string<C>& system_id,
               flags f,
               const properties<C>& p)
        {
          xml::auto_initializer init ((f & flags::dont_initialize) == 0);
          xml::sax::std_input_source isrc (is, system_id);
          parse (isrc, f, p);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               const std::basic_string<C>& system_id,
               xml::error_handler<C>& eh,
               flags f,
               const properties<C>& p)
        {
          xml::auto_initializer init ((f & flags::dont_initialize) == 0);
          xml::sax::std_input_source isrc (is, system_id);
          parse (isrc, eh, f, p);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               const std::basic_string<C>& system_id,
               xercesc::ErrorHandler& eh,
               flags f,
               const properties<C>& p)
        {
          xml::sax::std_input_source isrc (is, system_id);
          parse (isrc, eh, f, p);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               const std::basic_string<C>& system_id,
               xercesc::SAX2XMLReader& sax,
               flags f,
               const properties<C>& p)
        {
          xml::sax::std_input_source isrc (is, system_id);
          parse (isrc, sax, f, p);
        }


        // parse (istream, system_id, public_id)
        //

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               const std::basic_string<C>& system_id,
               const std::basic_string<C>& public_id,
               flags f,
               const properties<C>& p)
        {
          xml::auto_initializer init ((f & flags::dont_initialize) == 0);
          xml::sax::std_input_source isrc (is, system_id, public_id);
          parse (isrc, f, p);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               const std::basic_string<C>& system_id,
               const std::basic_string<C>& public_id,
               xml::error_handler<C>& eh,
               flags f,
               const properties<C>& p)
        {
          xml::auto_initializer init ((f & flags::dont_initialize) == 0);
          xml::sax::std_input_source isrc (is, system_id, public_id);
          parse (isrc, eh, f, p);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               const std::basic_string<C>& system_id,
               const std::basic_string<C>& public_id,
               xercesc::ErrorHandler& eh,
               flags f,
               const properties<C>& p)
        {
          xml::sax::std_input_source isrc (is, system_id, public_id);
          parse (isrc, eh, f, p);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               const std::basic_string<C>& system_id,
               const std::basic_string<C>& public_id,
               xercesc::SAX2XMLReader& sax,
               flags f,
               const properties<C>& p)
        {
          xml::sax::std_input_source isrc (is, system_id, public_id);
          parse (isrc, sax, f, p);
        }


        // parse (InputSource)
        //


        template <typename C>
        void document<C>::
        parse (const xercesc::InputSource& is,
               flags f,
               const properties<C>& p)
        {
          error_handler<C> eh;
          xml::sax::bits::error_handler_proxy<C> eh_proxy (eh);
          std::auto_ptr<xercesc::SAX2XMLReader> sax (create_sax_ (f, p));

          parse (is, eh_proxy, *sax, f, p);

          eh.throw_if_failed ();
        }

        template <typename C>
        void document<C>::
        parse (const xercesc::InputSource& is,
               xml::error_handler<C>& eh,
               flags f,
               const properties<C>& p)
        {
          xml::sax::bits::error_handler_proxy<C> eh_proxy (eh);
          std::auto_ptr<xercesc::SAX2XMLReader> sax (create_sax_ (f, p));

          parse (is, eh_proxy, *sax, f, p);

          if (eh_proxy.failed ())
            throw parsing<C> ();
        }

        template <typename C>
        void document<C>::
        parse (const xercesc::InputSource& is,
               xercesc::ErrorHandler& eh,
               flags f,
               const properties<C>& p)
        {
          xml::sax::bits::error_handler_proxy<C> eh_proxy (eh);
          std::auto_ptr<xercesc::SAX2XMLReader> sax (create_sax_ (f, p));

          parse (is, eh_proxy, *sax, f, p);

          if (eh_proxy.failed ())
            throw parsing<C> ();
        }


        template <typename C>
        void document<C>::
        parse (const xercesc::InputSource& is,
               xercesc::SAX2XMLReader& sax,
               flags f,
               const properties<C>& p)
        {
          // If there is no error handler, then fall back on the default
          // implementation.
          //
          xercesc::ErrorHandler* eh (sax.getErrorHandler ());

          if (eh)
          {
            xml::sax::bits::error_handler_proxy<C> eh_proxy (*eh);

            parse (is, eh_proxy, sax, f, p);

            if (eh_proxy.failed ())
              throw parsing<C> ();
          }
          else
          {
            error_handler<C> fallback_eh;
            xml::sax::bits::error_handler_proxy<C> eh_proxy (fallback_eh);

            parse (is, eh_proxy, sax, f, p);

            fallback_eh.throw_if_failed ();
          }
        }

        namespace Bits
        {
          struct ErrorHandlingController
          {
            ErrorHandlingController (xercesc::SAX2XMLReader& sax,
                                     xercesc::ErrorHandler& eh)
                : sax_ (sax), eh_ (sax_.getErrorHandler ())
            {
              sax_.setErrorHandler (&eh);
            }

            ~ErrorHandlingController ()
            {
              sax_.setErrorHandler (eh_);
            }

          private:
            xercesc::SAX2XMLReader& sax_;
            xercesc::ErrorHandler* eh_;
          };

          struct ContentHandlingController
          {
            ContentHandlingController (xercesc::SAX2XMLReader& sax,
                                       xercesc::ContentHandler& ch)
                : sax_ (sax), ch_ (sax_.getContentHandler ())
            {
              sax_.setContentHandler (&ch);
            }

            ~ContentHandlingController ()
            {
              sax_.setContentHandler (ch_);
            }

          private:
            xercesc::SAX2XMLReader& sax_;
            xercesc::ContentHandler* ch_;
          };
        };

        template <typename C>
        void document<C>::
        parse (const std::basic_string<C>& uri,
               xercesc::ErrorHandler& eh,
               xercesc::SAX2XMLReader& sax,
               flags,
               const properties<C>&)
        {
          event_router<C> router (*this, polymorphic_);

          Bits::ErrorHandlingController ehc (sax, eh);
          Bits::ContentHandlingController chc (sax, router);

          try
          {
            sax.parse (xml::string (uri).c_str ());
          }
          catch (const schema_exception<C>& e)
          {
            xml::string id (e.id ());

            xercesc::SAXParseException se (
              xml::string (e.message ()).c_str (),
              id.c_str (),
              id.c_str (),
#if _XERCES_VERSION >= 30000
              static_cast<XMLFileLoc> (e.line ()),
              static_cast<XMLFileLoc> (e.column ())
#else
              static_cast<XMLSSize_t> (e.line ()),
              static_cast<XMLSSize_t> (e.column ())
#endif
            );

            eh.fatalError (se);
          }
        }

        template <typename C>
        void document<C>::
        parse (const xercesc::InputSource& is,
               xercesc::ErrorHandler& eh,
               xercesc::SAX2XMLReader& sax,
               flags,
               const properties<C>&)
        {
          event_router<C> router (*this, polymorphic_);

          Bits::ErrorHandlingController controller (sax, eh);
          Bits::ContentHandlingController chc (sax, router);

          try
          {
            sax.parse (is);
          }
          catch (const schema_exception<C>& e)
          {
            xml::string id (e.id ());

            xercesc::SAXParseException se (
              xml::string (e.message ()).c_str (),
              id.c_str (),
              id.c_str (),
#if _XERCES_VERSION >= 30000
              static_cast<XMLFileLoc> (e.line ()),
              static_cast<XMLFileLoc> (e.column ())
#else
              static_cast<XMLSSize_t> (e.line ()),
              static_cast<XMLSSize_t> (e.column ())
#endif
            );

            eh.fatalError (se);
          }
        }


        template <typename C>
        std::auto_ptr<xercesc::SAX2XMLReader> document<C>::
        create_sax_ (flags f, const properties<C>& p)
        {
          // HP aCC cannot handle using namespace xercesc;
          //
          using xercesc::SAX2XMLReader;
          using xercesc::XMLReaderFactory;
          using xercesc::XMLUni;

          std::auto_ptr<SAX2XMLReader> sax (
            XMLReaderFactory::createXMLReader ());

          sax->setFeature (XMLUni::fgSAX2CoreNameSpaces, true);
          sax->setFeature (XMLUni::fgSAX2CoreNameSpacePrefixes, true);
          sax->setFeature (XMLUni::fgXercesValidationErrorAsFatal, true);

          if (f & flags::dont_validate)
          {
            sax->setFeature (XMLUni::fgSAX2CoreValidation, false);
            sax->setFeature (XMLUni::fgXercesSchema, false);
            sax->setFeature (XMLUni::fgXercesSchemaFullChecking, false);
          }
          else
          {
            sax->setFeature (XMLUni::fgSAX2CoreValidation, true);
            sax->setFeature (XMLUni::fgXercesSchema, true);

            // Xerces-C++ 3.1.0 is the first version with working multi import
            // support.
            //
#if _XERCES_VERSION >= 30100
            if (!(f & flags::no_multiple_imports))
              sax->setFeature (XMLUni::fgXercesHandleMultipleImports, true);
#endif
            // This feature checks the schema grammar for additional
            // errors. We most likely do not need it when validating
            // instances (assuming the schema is valid).
            //
            sax->setFeature (XMLUni::fgXercesSchemaFullChecking, false);
          }

          // Transfer properies if any.
          //

          if (!p.schema_location ().empty ())
          {
            xml::string sl (p.schema_location ());
            const void* v (sl.c_str ());

            sax->setProperty (
              XMLUni::fgXercesSchemaExternalSchemaLocation,
              const_cast<void*> (v));
          }

          if (!p.no_namespace_schema_location ().empty ())
          {
            xml::string sl (p.no_namespace_schema_location ());
            const void* v (sl.c_str ());

            sax->setProperty (
              XMLUni::fgXercesSchemaExternalNoNameSpaceSchemaLocation,
              const_cast<void*> (v));
          }

          return sax;
        }

        // event_router
        //
        template <typename C>
        event_router<C>::
        event_router (cxx::parser::document<C>& consumer, bool polymorphic)
            : loc_ (0), consumer_ (consumer), polymorphic_ (polymorphic)
        {
        }

        template <typename C>
        void event_router<C>::
        setDocumentLocator (const xercesc::Locator* const loc)
        {
          loc_ = loc;
        }

        template <typename C>
        void event_router<C>::
        startElement(const XMLCh* const uri,
                     const XMLCh* const lname,
                     const XMLCh* const /*qname*/,
                     const xercesc::Attributes& attributes)
        {
          typedef std::basic_string<C> string;

          {
            last_valid_ = true;
            last_ns_ = xml::transcode<C> (uri);
            last_name_ = xml::transcode<C> (lname);

            // Without this explicit construction IBM XL C++ complains
            // about ro_string's copy ctor being private even though the
            // temporary has been eliminated. Note that we cannot
            // eliminate ns, name and value since ro_string does not make
            // a copy.
            //
            ro_string<C> ro_ns (last_ns_);
            ro_string<C> ro_name (last_name_);

            if (!polymorphic_)
            {
              try
              {
                consumer_.start_element (ro_ns, ro_name, 0);
              }
              catch (schema_exception<C>& e)
              {
                set_location (e);
                throw;
              }
            }
            else
            {
              // Search for the xsi:type attribute.
              //
              int i (attributes.getIndex (
                       xercesc::SchemaSymbols::fgURI_XSI,
                       xercesc::SchemaSymbols::fgXSI_TYPE));

              if (i == -1)
              {
                try
                {
                  consumer_.start_element (ro_ns, ro_name, 0);
                }
                catch (schema_exception<C>& e)
                {
                  set_location (e);
                  throw;
                }
              }
              else
              {
                try
                {
                  // @@ Probably need proper QName validation.
                  //
                  // Get the qualified type name and try to resolve it.
                  //
                  string qn (xml::transcode<C> (attributes.getValue (i)));

                  ro_string<C> tp, tn;
                  typename string::size_type pos (qn.find (C (':')));

                  if (pos != string::npos)
                  {
                    tp.assign (qn.c_str (), pos);
                    tn.assign (qn.c_str () + pos + 1);

                    if (tp.empty ())
                      throw dynamic_type<C> (qn);
                  }
                  else
                    tn.assign (qn);

                  if (tn.empty ())
                    throw dynamic_type<C> (qn);

                  // Search our namespace declaration stack. Sun CC 5.7
                  // blows if we use const_reverse_iterator.
                  //
                  ro_string<C> tns;
                  for (typename ns_decls::reverse_iterator
                         it (ns_decls_.rbegin ()), e (ns_decls_.rend ());
                       it != e; ++it)
                  {
                    if (it->prefix == tp)
                    {
                      tns.assign (it->ns);
                      break;
                    }
                  }

                  if (!tp.empty () && tns.empty ())
                  {
                    // The 'xml' prefix requires special handling.
                    //
                    if (tp == xml::bits::xml_prefix<C> ())
                      tns.assign (xml::bits::xml_namespace<C> ());
                    else
                      throw dynamic_type<C> (qn);
                  }

                  // Construct the compound type id.
                  //
                  string id (tn.data (), tn.size ());

                  if (!tns.empty ())
                  {
                    id += C (' ');
                    id.append (tns.data (), tns.size ());
                  }

                  ro_string<C> ro_id (id);
                  consumer_.start_element (ro_ns, ro_name, &ro_id);
                }
                catch (schema_exception<C>& e)
                {
                  set_location (e);
                  throw;
                }
              }
            }
          }

#if _XERCES_VERSION >= 30000
          for (XMLSize_t i (0), end (attributes.getLength()); i < end; ++i)
#else
          for (unsigned int i (0), end (attributes.getLength()); i < end; ++i)
#endif
          {
            string ns (xml::transcode<C> (attributes.getURI (i)));
            string name (xml::transcode<C> (attributes.getLocalName (i)));
            string value (xml::transcode<C> (attributes.getValue (i)));

            // Without this explicit construction IBM XL C++ complains
            // about ro_string's copy ctor being private even though the
            // temporary has been eliminated. Note that we cannot
            // eliminate ns, name and value since ro_string does not make
            // a copy.
            //
            ro_string<C> ro_ns (ns);
            ro_string<C> ro_name (name);
            ro_string<C> ro_value (value);

            try
            {
              consumer_.attribute (ro_ns, ro_name, ro_value);
            }
            catch (schema_exception<C>& e)
            {
              set_location (e);
              throw;
            }
          }
        }

        template <typename C>
        void event_router<C>::
        endElement(const XMLCh* const uri,
                   const XMLCh* const lname,
                   const XMLCh* const /*qname*/)
        {
          typedef std::basic_string<C> string;

          try
          {
            // Without this explicit construction IBM XL C++ complains
            // about ro_string's copy ctor being private even though the
            // temporary has been eliminated. Note that we cannot
            // eliminate ns, name and value since ro_string does not make
            // a copy.
            //
            if (last_valid_)
            {
              last_valid_ = false;
              ro_string<C> ro_ns (last_ns_);
              ro_string<C> ro_name (last_name_);

              consumer_.end_element (ro_ns, ro_name);
            }
            else
            {
              string ns (xml::transcode<C> (uri));
              string name (xml::transcode<C> (lname));

              ro_string<C> ro_ns (ns);
              ro_string<C> ro_name (name);

              consumer_.end_element (ro_ns, ro_name);
            }
          }
          catch (schema_exception<C>& e)
          {
            set_location (e);
            throw;
          }
        }

        template <typename C>
        void event_router<C>::
#if _XERCES_VERSION >= 30000
        characters (const XMLCh* const s, const XMLSize_t n)
#else
        characters (const XMLCh* const s, const unsigned int n)
#endif
        {
          typedef std::basic_string<C> string;

          if (n != 0)
          {
            string str (xml::transcode<C> (s, n));

            // Without this explicit construction IBM XL C++ complains
            // about ro_string's copy ctor being private even though the
            // temporary has been eliminated. Note that we cannot
            // eliminate str since ro_string does not make a copy.
            //
            ro_string<C> ro_str (str);

            try
            {
              consumer_.characters (ro_str);
            }
            catch (schema_exception<C>& e)
            {
              set_location (e);
              throw;
            }
          }
        }

        template <typename C>
        void event_router<C>::
        startPrefixMapping (const XMLCh* const prefix,
                            const XMLCh* const uri)
        {
          if (polymorphic_)
          {
            typedef std::basic_string<C> string;

            string p (xml::transcode<C> (prefix));
            string ns (xml::transcode<C> (uri));

            ns_decls_.push_back (ns_decl (p, ns));
          }
        }

        template <typename C>
        void event_router<C>::
        endPrefixMapping (const XMLCh* const prefix)
        {
          if (polymorphic_)
          {
            typedef std::basic_string<C> string;

            string p (xml::transcode<C> (prefix));

            // Here we assume the prefixes are removed in the reverse
            // order of them being added. This appears to how every
            // sensible implementation works.
            //
            assert (ns_decls_.back ().prefix == p);

            ns_decls_.pop_back ();
          }
        }

        template <typename C>
        void event_router<C>::
        set_location (schema_exception<C>& e)
        {
          if (loc_ != 0)
          {
            const XMLCh* id (loc_->getPublicId ());

            if (id == 0)
              id = loc_->getSystemId ();

            if (id != 0)
              e.id (xml::transcode<C> (id));

#if _XERCES_VERSION >= 30000
            e.line (static_cast<unsigned long> (loc_->getLineNumber ()));
            e.column (static_cast<unsigned long> (loc_->getColumnNumber ()));
#else
            XMLSSize_t l (loc_->getLineNumber ());
            XMLSSize_t c (loc_->getColumnNumber ());

            e.line (l == -1 ? 0 : static_cast<unsigned long> (l));
            e.column (c == -1 ? 0: static_cast<unsigned long> (c));
#endif
          }
        }
      }
    }
  }
}
