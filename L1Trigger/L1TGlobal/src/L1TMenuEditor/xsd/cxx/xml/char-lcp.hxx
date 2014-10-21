// file      : xsd/cxx/xml/char-lcp.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_TRANSCODER
#define XSD_CXX_XML_TRANSCODER
#define XSD_CXX_XML_TRANSCODER_CHAR_LCP

#include <string>
#include <cstddef> // std::size_t

#include <xercesc/util/XercesDefs.hpp> // XMLCh

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      // UTF-16 to/from Xerces-C++ local code page (LCP) transcoder.
      //
      // Note that this transcoder has a custom interface due to Xerces-C++
      // idiosyncrasies. Don't use it as a base for your custom transcoder.
      //
      template <typename C>
      struct char_lcp_transcoder
      {
        static std::basic_string<C>
        to (const XMLCh* s);

        static std::basic_string<C>
        to (const XMLCh* s, std::size_t length);

        static XMLCh*
        from (const C* s);
      };

      typedef char_lcp_transcoder<char> char_transcoder;
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/char-lcp.txx>

#else
#  ifndef XSD_CXX_XML_TRANSCODER_CHAR_LCP
     //
     // If you get this error, it usually means that either you compiled
     // your schemas with different --char-encoding values or you included
     // some of the libxsd headers (e.g., xsd/cxx/xml/string.hxx) directly
     // without first including the correct xsd/cxx/xml/char-*.hxx header.
     //
#    error conflicting character encoding detected
#  endif
#endif // XSD_CXX_XML_TRANSCODER
