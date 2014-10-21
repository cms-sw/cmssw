// file      : xsd/cxx/xml/dom/serialization-header.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_DOM_SERIALIZATION_HEADER_HXX
#define XSD_CXX_XML_DOM_SERIALIZATION_HEADER_HXX

#include <map>
#include <string>

#include <xercesc/dom/DOMElement.hpp>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace dom
      {
        // Find an existing prefix or establish a new one. Try to use
        // hint if provided and available.
        //
        template <typename C>
        std::basic_string<C>
        prefix (const C* ns, xercesc::DOMElement&, const C* hint = 0);

        template <typename C>
        inline std::basic_string<C>
        prefix (const std::basic_string<C>& ns,
                xercesc::DOMElement& e,
                const C* hint = 0)
        {
          return prefix (ns.c_str (), e, hint);
        }

        //
        //
        template <typename C>
        void
        clear (xercesc::DOMElement&);

        //
        //
        template <typename C>
        class namespace_info
        {
	public:
          typedef std::basic_string<C> string;

          namespace_info ()
          {
          }

          namespace_info (const string& name_, const string& schema_)
              : name (name_),
                schema (schema_)
          {
          }

          std::basic_string<C> name;
          std::basic_string<C> schema;
        };


        // Map of namespace prefix to namespace_info.
        //
        template <typename C>
        class namespace_infomap:
          public std::map<std::basic_string<C>, namespace_info<C> >
        {
        };
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/serialization-header.txx>

#endif  // XSD_CXX_XML_DOM_SERIALIZATION_HEADER_HXX
