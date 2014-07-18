// file      : xsd/cxx/xml/dom/serialization-source.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <xercesc/util/XMLUni.hpp>     // xercesc::fg*
#include <xercesc/util/XMLUniDefs.hpp> // chLatin_L, etc
#include <xercesc/validators/schema/SchemaSymbols.hpp>

#if _XERCES_VERSION >= 30000
#  include <xercesc/dom/DOMLSOutput.hpp>
#  include <xercesc/dom/DOMLSSerializer.hpp>
#else
#  include <xercesc/dom/DOMWriter.hpp>
#endif
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/bits/literals.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/bits/error-handler-proxy.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace dom
      {
        //
        //
        template <typename C>
        xercesc::DOMAttr&
        create_attribute (const C* name, xercesc::DOMElement& parent)
        {
          xercesc::DOMDocument* doc (parent.getOwnerDocument ());
          xercesc::DOMAttr* a (doc->createAttribute (string (name).c_str ()));
          parent.setAttributeNode (a);
          return *a;
        }

        template <typename C>
        xercesc::DOMAttr&
        create_attribute (const C* name,
                          const C* ns,
                          xercesc::DOMElement& parent)
        {
          if (ns[0] == C ('\0'))
            return create_attribute (name, parent);

          xercesc::DOMDocument* doc (parent.getOwnerDocument ());

          xercesc::DOMAttr* a;
          std::basic_string<C> p (prefix<C> (ns, parent));

          if (!p.empty ())
          {
            p += ':';
            p += name;
            a = doc->createAttributeNS (string (ns).c_str (),
                                        string (p).c_str ());
          }
          else
            a = doc->createAttributeNS (string (ns).c_str (),
                                        string (name).c_str ());

          parent.setAttributeNodeNS (a);
          return *a;
        }

        template <typename C>
        xercesc::DOMElement&
        create_element (const C* name, xercesc::DOMElement& parent)
        {
          xercesc::DOMDocument* doc (parent.getOwnerDocument ());
          xercesc::DOMElement* e (doc->createElement (string (name).c_str ()));
          parent.appendChild (e);
          return *e;
        }

        template <typename C>
        xercesc::DOMElement&
        create_element (const C* name,
                        const C* ns,
                        xercesc::DOMElement& parent)
        {
          if (ns[0] == C ('\0'))
            return create_element (name, parent);

          xercesc::DOMDocument* doc (parent.getOwnerDocument ());

          xercesc::DOMElement* e;
          std::basic_string<C> p (prefix<C> (ns, parent));

          if (!p.empty ())
          {
            p += ':';
            p += name;
            e = doc->createElementNS (string (ns).c_str (),
                                      string (p).c_str ());
          }
          else
            e = doc->createElementNS (string (ns).c_str (),
                                      string (name).c_str ());

          parent.appendChild (e);
          return *e;
        }


        //
        //
        template <typename C>
        auto_ptr<xercesc::DOMDocument>
        serialize (const std::basic_string<C>& el,
                   const std::basic_string<C>& ns,
                   const namespace_infomap<C>& map,
                   unsigned long)
        {
          // HP aCC cannot handle using namespace xercesc;
          //
          using xercesc::DOMImplementationRegistry;
          using xercesc::DOMImplementation;
          using xercesc::DOMDocument;
          using xercesc::DOMElement;

          //
          //
          typedef std::basic_string<C> string;
          typedef namespace_infomap<C> infomap;
          typedef typename infomap::const_iterator infomap_iterator;

          C colon (':'), space (' ');

          string prefix;

          if (!ns.empty ())
          {
            infomap_iterator i (map.begin ()), e (map.end ());

            for ( ;i != e; ++i)
            {
              if (i->second.name == ns)
              {
                prefix = i->first;
                break;
              }
            }

            // Since this is the first namespace in document we don't
            // need to worry about conflicts.
            //
            if (i == e)
              prefix = xml::bits::first_prefix<C> ();
          }

          const XMLCh ls[] = {xercesc::chLatin_L,
                              xercesc::chLatin_S,
                              xercesc::chNull};

          DOMImplementation* impl (
            DOMImplementationRegistry::getDOMImplementation (ls));

          auto_ptr<DOMDocument> doc (
            impl->createDocument (
              (ns.empty () ? 0 : xml::string (ns).c_str ()),
              xml::string ((prefix.empty ()
                            ? el
                            : prefix + colon + el)).c_str (),
              0));

          DOMElement* root (doc->getDocumentElement ());

          // Check if we need to provide xsi mapping.
          //
          bool xsi (false);
          string xsi_prefix;
          string xmlns_prefix (xml::bits::xmlns_prefix<C> ());

          for (infomap_iterator i (map.begin ()), e (map.end ()); i != e; ++i)
          {
            if (!i->second.schema.empty ())
            {
              xsi = true;
              break;
            }
          }

          // Check if we were told to provide xsi mapping.
          //
          if (xsi)
          {
            for (infomap_iterator i (map.begin ()), e (map.end ());
                 i != e;
                 ++i)
            {
              if (i->second.name == xml::bits::xsi_namespace<C> ())
              {
                xsi_prefix = i->first;
                xsi = false;
                break;
              }
            }
          }

          // Create user-defined mappings.
          //
          for (infomap_iterator i (map.begin ()), e (map.end ()); i != e; ++i)
          {
            if (i->first.empty ())
            {
              // Empty prefix.
              //
              if (!i->second.name.empty ())
                root->setAttributeNS (
                  xercesc::XMLUni::fgXMLNSURIName,
                  xml::string (xmlns_prefix).c_str (),
                  xml::string (i->second.name).c_str ());
            }
            else
            {
              root->setAttributeNS (
                xercesc::XMLUni::fgXMLNSURIName,
                xml::string (xmlns_prefix + colon + i->first).c_str (),
                xml::string (i->second.name).c_str ());
            }
          }

          // If we were not told to provide xsi mapping but we need it
          // then we will have to add it ourselves.
          //
          if (xsi)
            xsi_prefix = dom::prefix (xml::bits::xsi_namespace<C> (),
                                      *root,
                                      xml::bits::xsi_prefix<C> ());

          // Create xsi:schemaLocation and xsi:noNamespaceSchemaLocation
          // attributes.
          //
          string schema_location;
          string no_namespace_schema_location;

          for (infomap_iterator i (map.begin ()), e (map.end ()); i != e; ++i)
          {
            if (!i->second.schema.empty ())
            {
              if (i->second.name.empty ())
              {
                if (!no_namespace_schema_location.empty ())
                  no_namespace_schema_location += space;

                no_namespace_schema_location += i->second.schema;
              }
              else
              {
                if (!schema_location.empty ())
                  schema_location += space;

                schema_location += i->second.name + space + i->second.schema;
              }
            }
          }

          if (!schema_location.empty ())
          {
            root->setAttributeNS (
              xercesc::SchemaSymbols::fgURI_XSI,
              xml::string (xsi_prefix + colon +
                           xml::bits::schema_location<C> ()).c_str (),
              xml::string (schema_location).c_str ());
          }

          if (!no_namespace_schema_location.empty ())
          {
            root->setAttributeNS (
              xercesc::SchemaSymbols::fgURI_XSI,
              xml::string (
                xsi_prefix + colon +
                xml::bits::no_namespace_schema_location<C> ()).c_str (),
              xml::string (no_namespace_schema_location).c_str ());
          }

          return doc;
        }


        template <typename C>
        bool
        serialize (xercesc::XMLFormatTarget& target,
                   const xercesc::DOMDocument& doc,
                   const std::basic_string<C>& encoding,
                   xercesc::DOMErrorHandler& eh,
                   unsigned long flags)
        {
          // HP aCC cannot handle using namespace xercesc;
          //
          using xercesc::DOMImplementationRegistry;
          using xercesc::DOMImplementation;
#if _XERCES_VERSION >= 30000
          using xercesc::DOMLSSerializer;
          using xercesc::DOMConfiguration;
          using xercesc::DOMLSOutput;
#else
          using xercesc::DOMWriter;
#endif
          using xercesc::XMLUni;

          const XMLCh ls[] = {xercesc::chLatin_L,
                              xercesc::chLatin_S,
                              xercesc::chNull};

          DOMImplementation* impl (
            DOMImplementationRegistry::getDOMImplementation (ls));

          bits::error_handler_proxy<C> ehp (eh);

#if _XERCES_VERSION >= 30000
          xml::dom::auto_ptr<DOMLSSerializer> writer (
            impl->createLSSerializer ());

          DOMConfiguration* conf (writer->getDomConfig ());

          conf->setParameter (XMLUni::fgDOMErrorHandler, &ehp);

          // Set some nice features if the serializer supports them.
          //
          if (conf->canSetParameter (
                XMLUni::fgDOMWRTDiscardDefaultContent, true))
            conf->setParameter (XMLUni::fgDOMWRTDiscardDefaultContent, true);

          if (!(flags & dont_pretty_print) &&
              conf->canSetParameter (XMLUni::fgDOMWRTFormatPrettyPrint, true))
            conf->setParameter (XMLUni::fgDOMWRTFormatPrettyPrint, true);

          // See if we need to write XML declaration.
          //
          if ((flags & no_xml_declaration) &&
              conf->canSetParameter (XMLUni::fgDOMXMLDeclaration, false))
            conf->setParameter (XMLUni::fgDOMXMLDeclaration, false);

          xml::dom::auto_ptr<DOMLSOutput> out (impl->createLSOutput ());

          out->setEncoding (xml::string (encoding).c_str ());
          out->setByteStream (&target);

          bool r (writer->write (&doc, out.get ()));
#else
          xml::dom::auto_ptr<DOMWriter> writer (impl->createDOMWriter ());

          writer->setErrorHandler (&ehp);
          writer->setEncoding (xml::string (encoding).c_str ());

          // Set some nice features if the serializer supports them.
          //
          if (writer->canSetFeature (
                XMLUni::fgDOMWRTDiscardDefaultContent, true))
            writer->setFeature (XMLUni::fgDOMWRTDiscardDefaultContent, true);

          if (!(flags & dont_pretty_print) &&
              writer->canSetFeature (XMLUni::fgDOMWRTFormatPrettyPrint, true))
            writer->setFeature (XMLUni::fgDOMWRTFormatPrettyPrint, true);

          // See if we need to write XML declaration.
          //
          if ((flags & no_xml_declaration) &&
              writer->canSetFeature (XMLUni::fgDOMXMLDeclaration, false))
            writer->setFeature (XMLUni::fgDOMXMLDeclaration, false);

          bool r (writer->writeNode (&target, doc));
#endif

          if (!r || ehp.failed ())
            return false;

          return true;
        }

        template <typename C>
        bool
        serialize (xercesc::XMLFormatTarget& target,
                   const xercesc::DOMDocument& doc,
                   const std::basic_string<C>& enconding,
                   error_handler<C>& eh,
                   unsigned long flags)
        {
          bits::error_handler_proxy<C> ehp (eh);
          return serialize (target, doc, enconding, ehp, flags);
        }
      }
    }
  }
}
