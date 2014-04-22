// file      : xsd/cxx/xml/dom/serialization-header.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <vector>
#include <sstream>
#include <cstddef> // std::size_t

#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>

#include <xercesc/util/XMLUni.hpp>    // xercesc::fg*
#include <xercesc/util/XMLString.hpp>
#include <xercesc/validators/schema/SchemaSymbols.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/bits/literals.hxx>

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
        std::basic_string<C>
        prefix (const C* ns, xercesc::DOMElement& e, const C* hint)
        {
          string xns (ns);

#if _XERCES_VERSION >= 30000
          const XMLCh* p (e.lookupPrefix (xns.c_str ()));
#else
          const XMLCh* p (e.lookupNamespacePrefix (xns.c_str (), false));
#endif
          if (p != 0)
            return transcode<C> (p);

          if (e.isDefaultNamespace (xns.c_str ()))
            return std::basic_string<C> ();

          // 'xml' prefix requires special handling and Xerces folks
          // refuse to handle this in DOM so I have to do it myself.
          //
          if (std::basic_string<C> (ns) == xml::bits::xml_namespace<C> ())
            return xml::bits::xml_prefix<C> ();

          // No prefix for this namespace. Will need to establish one.
          //
          std::basic_string<C> prefix;

          if (hint != 0 &&
              e.lookupNamespaceURI (xml::string (hint).c_str ()) == 0)
          {
            prefix = hint;
          }
          else
          {
            for (unsigned long n (1);; ++n)
            {
              // Make finding the first few prefixes fast.
              //
              switch (n)
              {
              case 1:
                {
                  prefix = xml::bits::first_prefix<C> ();
                  break;
                }
              case 2:
                {
                  prefix = xml::bits::second_prefix<C> ();
                  break;
                }
              case 3:
                {
                  prefix = xml::bits::third_prefix<C> ();
                  break;
                }
              case 4:
                {
                  prefix = xml::bits::fourth_prefix<C> ();
                  break;
                }
              case 5:
                {
                  prefix = xml::bits::fifth_prefix<C> ();
                  break;
                }
              default:
                {
                  std::basic_ostringstream<C> ostr;
                  ostr << C ('p') << n;
                  prefix = ostr.str ();
                  break;
                }
              }

              if (e.lookupNamespaceURI (xml::string (prefix).c_str ()) == 0)
                break;
            }
          }

          std::basic_string<C> name (xml::bits::xmlns_prefix<C> ());
          name += C(':');
          name += prefix;

          e.setAttributeNS (
            xercesc::XMLUni::fgXMLNSURIName,
            xml::string (name).c_str (),
            xns.c_str ());

          return prefix;
        }

        //
        //
        template <typename C>
        void
        clear (xercesc::DOMElement& e)
        {
          // HP aCC cannot handle using namespace xercesc;
          //
          using xercesc::DOMNode;
          using xercesc::DOMAttr;
          using xercesc::DOMNamedNodeMap;
          using xercesc::XMLString;
          using xercesc::SchemaSymbols;

          // Remove child nodes.
          //
          while (xercesc::DOMNode* n = e.getFirstChild ())
          {
            e.removeChild (n);
            n->release ();
          }

          // Remove attributes.
          //
          DOMNamedNodeMap* att_map (e.getAttributes ());
          XMLSize_t n (att_map->getLength ());

          if (n != 0)
          {
            std::vector<DOMAttr*> atts;

            // Collect all attributes to be removed while filtering
            // out special cases (xmlns & xsi).
            //
            for (XMLSize_t i (0); i != n; ++i)
            {
              DOMAttr* a (static_cast<DOMAttr*> (att_map->item (i)));
              const XMLCh* ns (a->getNamespaceURI ());

              if (ns != 0)
              {
                if (XMLString::equals (ns, xercesc::XMLUni::fgXMLNSURIName))
                  continue;

                if (XMLString::equals (ns, SchemaSymbols::fgURI_XSI))
                {
                  const XMLCh* name (a->getLocalName ());

                  if (XMLString::equals (
                        name, SchemaSymbols::fgXSI_SCHEMALOCACTION) ||
                      XMLString::equals (
                        name, SchemaSymbols::fgXSI_NONAMESPACESCHEMALOCACTION))
                    continue;
                }
              }

              atts.push_back (a);
            }

            for (std::vector<DOMAttr*>::iterator i (atts.begin ()),
                   end (atts.end ()); i != end; ++i)
            {
              e.removeAttributeNode (*i);
              (*i)->release ();
            }
          }
        }
      }
    }
  }
}
