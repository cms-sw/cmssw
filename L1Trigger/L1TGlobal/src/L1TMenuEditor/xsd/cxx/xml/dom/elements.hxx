// file      : xsd/cxx/xml/dom/elements.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_DOM_ELEMENTS_HXX
#define XSD_CXX_XML_DOM_ELEMENTS_HXX

#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMElement.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/qualified-name.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace dom
      {
        template <typename C>
        qualified_name<C>
        name (const xercesc::DOMAttr&);

        template <typename C>
        qualified_name<C>
        name (const xercesc::DOMElement&);
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/elements.txx>

#endif // XSD_CXX_XML_DOM_ELEMENTS_HXX
