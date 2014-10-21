// file      : xsd/cxx/xml/dom/wildcard-source.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <xercesc/util/XMLUniDefs.hpp> // chLatin_L, etc

#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>

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
        create_document ()
        {
          const XMLCh ls[] = {xercesc::chLatin_L,
                              xercesc::chLatin_S,
                              xercesc::chNull};

          // Get an implementation of the Load-Store (LS) interface.
          //
          xercesc::DOMImplementation* impl (
            xercesc::DOMImplementationRegistry::getDOMImplementation (ls));

          return xml::dom::auto_ptr<xercesc::DOMDocument> (
            impl->createDocument ());
        }
      }
    }
  }
}
