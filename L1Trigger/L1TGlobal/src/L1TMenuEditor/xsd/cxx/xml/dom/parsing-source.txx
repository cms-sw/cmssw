// file      : xsd/cxx/xml/dom/parsing-source.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#if _XERCES_VERSION >= 30000
#  include <xercesc/dom/DOMLSParser.hpp>
#  include <xercesc/dom/DOMLSException.hpp>
#else
#  include <xercesc/dom/DOMBuilder.hpp>
#endif
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>

#include <xercesc/util/XMLUni.hpp>     // xercesc::fg*
#include <xercesc/util/XMLUniDefs.hpp> // chLatin_L, etc

#include <xercesc/framework/Wrapper4InputSource.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/bits/error-handler-proxy.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace dom
      {
        // parser
        //
        template <typename C>
        parser<C>::
        parser (const xercesc::DOMElement& e, bool ep, bool ap)
            : element_ (e),
              next_element_ (0),
              a_ (0),
              ai_ (0)
        {
          using xercesc::DOMNode;

          if (ep)
          {
            for (next_element_ = e.getFirstChild ();
                 next_element_ != 0 &&
                   next_element_->getNodeType () != DOMNode::ELEMENT_NODE;
                 next_element_ = next_element_->getNextSibling ()) /*noop*/;
          }

          if (ap)
          {
            a_ = e.getAttributes ();
            as_ = a_->getLength ();
          }
        }

        template <typename C>
        void parser<C>::
        next_element ()
        {
          using xercesc::DOMNode;

          for (next_element_ = next_element_->getNextSibling ();
               next_element_ != 0 &&
                 next_element_->getNodeType () != DOMNode::ELEMENT_NODE;
               next_element_ = next_element_->getNextSibling ())/*noop*/;
        }

        // parse()
        //
        template <typename C>
        xml::dom::auto_ptr<xercesc::DOMDocument>
        parse (xercesc::InputSource& is,
               error_handler<C>& eh,
               const properties<C>& prop,
               unsigned long flags)
        {
          bits::error_handler_proxy<C> ehp (eh);
          return xml::dom::parse (is, ehp, prop, flags);
        }

        template <typename C>
        auto_ptr<xercesc::DOMDocument>
        parse (xercesc::InputSource& is,
               xercesc::DOMErrorHandler& eh,
               const properties<C>& prop,
               unsigned long flags)
        {
          // HP aCC cannot handle using namespace xercesc;
          //
          using xercesc::DOMImplementationRegistry;
          using xercesc::DOMImplementationLS;
          using xercesc::DOMImplementation;
          using xercesc::DOMDocument;
#if _XERCES_VERSION >= 30000
          using xercesc::DOMLSParser;
          using xercesc::DOMConfiguration;
#else
          using xercesc::DOMBuilder;
#endif

          using xercesc::Wrapper4InputSource;
          using xercesc::XMLUni;


          // Instantiate the DOM parser.
          //
          const XMLCh ls_id[] = {xercesc::chLatin_L,
                                 xercesc::chLatin_S,
                                 xercesc::chNull};

          // Get an implementation of the Load-Store (LS) interface.
          //
          DOMImplementation* impl (
            DOMImplementationRegistry::getDOMImplementation (ls_id));

#if _XERCES_VERSION >= 30000
          auto_ptr<DOMLSParser> parser (
            impl->createLSParser (DOMImplementationLS::MODE_SYNCHRONOUS, 0));

          DOMConfiguration* conf (parser->getDomConfig ());

          // Discard comment nodes in the document.
          //
          conf->setParameter (XMLUni::fgDOMComments, false);

          // Enable datatype normalization.
          //
          conf->setParameter (XMLUni::fgDOMDatatypeNormalization, true);

          // Do not create EntityReference nodes in the DOM tree. No
          // EntityReference nodes will be created, only the nodes
          // corresponding to their fully expanded substitution text
          // will be created.
          //
          conf->setParameter (XMLUni::fgDOMEntities, false);

          // Perform namespace processing.
          //
          conf->setParameter (XMLUni::fgDOMNamespaces, true);

          // Do not include ignorable whitespace in the DOM tree.
          //
          conf->setParameter (XMLUni::fgDOMElementContentWhitespace, false);

          if (flags & dont_validate)
          {
            conf->setParameter (XMLUni::fgDOMValidate, false);
            conf->setParameter (XMLUni::fgXercesSchema, false);
            conf->setParameter (XMLUni::fgXercesSchemaFullChecking, false);
          }
          else
          {
            conf->setParameter (XMLUni::fgDOMValidate, true);
            conf->setParameter (XMLUni::fgXercesSchema, true);

            // Xerces-C++ 3.1.0 is the first version with working multi import
            // support.
            //
#if _XERCES_VERSION >= 30100
            if (!(flags & no_muliple_imports))
              conf->setParameter (XMLUni::fgXercesHandleMultipleImports, true);
#endif

            // This feature checks the schema grammar for additional
            // errors. We most likely do not need it when validating
            // instances (assuming the schema is valid).
            //
            conf->setParameter (XMLUni::fgXercesSchemaFullChecking, false);
          }

          // We will release DOM ourselves.
          //
          conf->setParameter (XMLUni::fgXercesUserAdoptsDOMDocument, true);


          // Transfer properies if any.
          //

          if (!prop.schema_location ().empty ())
          {
            xml::string sl (prop.schema_location ());
            const void* v (sl.c_str ());

            conf->setParameter (
              XMLUni::fgXercesSchemaExternalSchemaLocation,
              const_cast<void*> (v));
          }

          if (!prop.no_namespace_schema_location ().empty ())
          {
            xml::string sl (prop.no_namespace_schema_location ());
            const void* v (sl.c_str ());

            conf->setParameter (
              XMLUni::fgXercesSchemaExternalNoNameSpaceSchemaLocation,
              const_cast<void*> (v));
          }

          // If external schema location was specified, disable loading
          // schemas via the schema location attributes in the document.
          //
#if _XERCES_VERSION >= 30100
          if (!prop.schema_location ().empty () ||
              !prop.no_namespace_schema_location ().empty ())
          {
            conf->setParameter (XMLUni::fgXercesLoadSchema, false);
          }
#endif
          // Set error handler.
          //
          bits::error_handler_proxy<C> ehp (eh);
          conf->setParameter (XMLUni::fgDOMErrorHandler, &ehp);

#else // _XERCES_VERSION >= 30000

          // Same as above but for Xerces-C++ 2 series.
          //
          auto_ptr<DOMBuilder> parser (
            impl->createDOMBuilder (DOMImplementationLS::MODE_SYNCHRONOUS, 0));

          parser->setFeature (XMLUni::fgDOMComments, false);
          parser->setFeature (XMLUni::fgDOMDatatypeNormalization, true);
          parser->setFeature (XMLUni::fgDOMEntities, false);
          parser->setFeature (XMLUni::fgDOMNamespaces, true);
          parser->setFeature (XMLUni::fgDOMWhitespaceInElementContent, false);

          if (flags & dont_validate)
          {
            parser->setFeature (XMLUni::fgDOMValidation, false);
            parser->setFeature (XMLUni::fgXercesSchema, false);
            parser->setFeature (XMLUni::fgXercesSchemaFullChecking, false);
          }
          else
          {
            parser->setFeature (XMLUni::fgDOMValidation, true);
            parser->setFeature (XMLUni::fgXercesSchema, true);
            parser->setFeature (XMLUni::fgXercesSchemaFullChecking, false);
          }

          parser->setFeature (XMLUni::fgXercesUserAdoptsDOMDocument, true);

          if (!prop.schema_location ().empty ())
          {
            xml::string sl (prop.schema_location ());
            const void* v (sl.c_str ());

            parser->setProperty (
              XMLUni::fgXercesSchemaExternalSchemaLocation,
              const_cast<void*> (v));
          }

          if (!prop.no_namespace_schema_location ().empty ())
          {
            xml::string sl (prop.no_namespace_schema_location ());
            const void* v (sl.c_str ());

            parser->setProperty (
              XMLUni::fgXercesSchemaExternalNoNameSpaceSchemaLocation,
              const_cast<void*> (v));
          }

          bits::error_handler_proxy<C> ehp (eh);
          parser->setErrorHandler (&ehp);

#endif // _XERCES_VERSION >= 30000

          xercesc::Wrapper4InputSource wrap (&is, false);

#if _XERCES_VERSION >= 30000
          auto_ptr<DOMDocument> doc;

          try
          {
            doc.reset (parser->parse (&wrap));
          }
          catch (const xercesc::DOMLSException&)
          {
          }
#else
          auto_ptr<DOMDocument> doc (parser->parse (wrap));
#endif
          if (ehp.failed ())
            doc.reset ();

          return doc;
        }

        template <typename C>
        xml::dom::auto_ptr<xercesc::DOMDocument>
        parse (const std::basic_string<C>& uri,
               error_handler<C>& eh,
               const properties<C>& prop,
               unsigned long flags)
        {
          bits::error_handler_proxy<C> ehp (eh);
          return xml::dom::parse (uri, ehp, prop, flags);
        }

        template <typename C>
        auto_ptr<xercesc::DOMDocument>
        parse (const std::basic_string<C>& uri,
               xercesc::DOMErrorHandler& eh,
               const properties<C>& prop,
               unsigned long flags)
        {
          // HP aCC cannot handle using namespace xercesc;
          //
          using xercesc::DOMImplementationRegistry;
          using xercesc::DOMImplementationLS;
          using xercesc::DOMImplementation;
          using xercesc::DOMDocument;
#if _XERCES_VERSION >= 30000
          using xercesc::DOMLSParser;
          using xercesc::DOMConfiguration;
#else
          using xercesc::DOMBuilder;
#endif
          using xercesc::XMLUni;


          // Instantiate the DOM parser.
          //
          const XMLCh ls_id[] = {xercesc::chLatin_L,
                                 xercesc::chLatin_S,
                                 xercesc::chNull};

          // Get an implementation of the Load-Store (LS) interface.
          //
          DOMImplementation* impl (
            DOMImplementationRegistry::getDOMImplementation (ls_id));

#if _XERCES_VERSION >= 30000
          auto_ptr<DOMLSParser> parser (
            impl->createLSParser(DOMImplementationLS::MODE_SYNCHRONOUS, 0));

          DOMConfiguration* conf (parser->getDomConfig ());

          // Discard comment nodes in the document.
          //
          conf->setParameter (XMLUni::fgDOMComments, false);

          // Enable datatype normalization.
          //
          conf->setParameter (XMLUni::fgDOMDatatypeNormalization, true);

          // Do not create EntityReference nodes in the DOM tree. No
          // EntityReference nodes will be created, only the nodes
          // corresponding to their fully expanded substitution text
          // will be created.
          //
          conf->setParameter (XMLUni::fgDOMEntities, false);

          // Perform namespace processing.
          //
          conf->setParameter (XMLUni::fgDOMNamespaces, true);

          // Do not include ignorable whitespace in the DOM tree.
          //
          conf->setParameter (XMLUni::fgDOMElementContentWhitespace, false);

          if (flags & dont_validate)
          {
            conf->setParameter (XMLUni::fgDOMValidate, false);
            conf->setParameter (XMLUni::fgXercesSchema, false);
            conf->setParameter (XMLUni::fgXercesSchemaFullChecking, false);
          }
          else
          {
            conf->setParameter (XMLUni::fgDOMValidate, true);
            conf->setParameter (XMLUni::fgXercesSchema, true);

            // Xerces-C++ 3.1.0 is the first version with working multi import
            // support.
            //
#if _XERCES_VERSION >= 30100
            if (!(flags & no_muliple_imports))
              conf->setParameter (XMLUni::fgXercesHandleMultipleImports, true);
#endif

            // This feature checks the schema grammar for additional
            // errors. We most likely do not need it when validating
            // instances (assuming the schema is valid).
            //
            conf->setParameter (XMLUni::fgXercesSchemaFullChecking, false);
          }

          // We will release DOM ourselves.
          //
          conf->setParameter (XMLUni::fgXercesUserAdoptsDOMDocument, true);


          // Transfer properies if any.
          //

          if (!prop.schema_location ().empty ())
          {
            xml::string sl (prop.schema_location ());
            const void* v (sl.c_str ());

            conf->setParameter (
              XMLUni::fgXercesSchemaExternalSchemaLocation,
              const_cast<void*> (v));
          }

          if (!prop.no_namespace_schema_location ().empty ())
          {
            xml::string sl (prop.no_namespace_schema_location ());
            const void* v (sl.c_str ());

            conf->setParameter (
              XMLUni::fgXercesSchemaExternalNoNameSpaceSchemaLocation,
              const_cast<void*> (v));
          }

          // If external schema location was specified, disable loading
          // schemas via the schema location attributes in the document.
          //
#if _XERCES_VERSION >= 30100
          if (!prop.schema_location ().empty () ||
              !prop.no_namespace_schema_location ().empty ())
          {
            conf->setParameter (XMLUni::fgXercesLoadSchema, false);
          }
#endif
          // Set error handler.
          //
          bits::error_handler_proxy<C> ehp (eh);
          conf->setParameter (XMLUni::fgDOMErrorHandler, &ehp);

#else // _XERCES_VERSION >= 30000

          // Same as above but for Xerces-C++ 2 series.
          //
          auto_ptr<DOMBuilder> parser (
            impl->createDOMBuilder(DOMImplementationLS::MODE_SYNCHRONOUS, 0));

          parser->setFeature (XMLUni::fgDOMComments, false);
          parser->setFeature (XMLUni::fgDOMDatatypeNormalization, true);
          parser->setFeature (XMLUni::fgDOMEntities, false);
          parser->setFeature (XMLUni::fgDOMNamespaces, true);
          parser->setFeature (XMLUni::fgDOMWhitespaceInElementContent, false);

          if (flags & dont_validate)
          {
            parser->setFeature (XMLUni::fgDOMValidation, false);
            parser->setFeature (XMLUni::fgXercesSchema, false);
            parser->setFeature (XMLUni::fgXercesSchemaFullChecking, false);
          }
          else
          {
            parser->setFeature (XMLUni::fgDOMValidation, true);
            parser->setFeature (XMLUni::fgXercesSchema, true);
            parser->setFeature (XMLUni::fgXercesSchemaFullChecking, false);
          }

          parser->setFeature (XMLUni::fgXercesUserAdoptsDOMDocument, true);

          if (!prop.schema_location ().empty ())
          {
            xml::string sl (prop.schema_location ());
            const void* v (sl.c_str ());

            parser->setProperty (
              XMLUni::fgXercesSchemaExternalSchemaLocation,
              const_cast<void*> (v));
          }

          if (!prop.no_namespace_schema_location ().empty ())
          {
            xml::string sl (prop.no_namespace_schema_location ());
            const void* v (sl.c_str ());

            parser->setProperty (
              XMLUni::fgXercesSchemaExternalNoNameSpaceSchemaLocation,
              const_cast<void*> (v));
          }

          bits::error_handler_proxy<C> ehp (eh);
          parser->setErrorHandler (&ehp);

#endif // _XERCES_VERSION >= 30000


#if _XERCES_VERSION >= 30000
          auto_ptr<DOMDocument> doc;

          try
          {
            doc.reset (parser->parseURI (string (uri).c_str ()));
          }
          catch (const xercesc::DOMLSException&)
          {
          }
#else
          auto_ptr<DOMDocument> doc (
            parser->parseURI (string (uri).c_str ()));
#endif

          if (ehp.failed ())
            doc.reset ();

          return doc;
        }
      }
    }
  }
}
