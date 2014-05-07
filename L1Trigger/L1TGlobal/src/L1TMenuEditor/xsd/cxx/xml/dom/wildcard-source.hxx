// file      : xsd/cxx/xml/dom/wildcard-source.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_DOM_WILDCARD_SOURCE_HXX
#define XSD_CXX_XML_DOM_WILDCARD_SOURCE_HXX

#include <xercesc/dom/DOMDocument.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/auto-ptr.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace dom
      {
        template <typename C>
        xml::dom::auto_ptr<xercesc::DOMDocument>
        create_document ();
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/wildcard-source.txx>

#endif // XSD_CXX_XML_DOM_WILDCARD_SOURCE_HXX
