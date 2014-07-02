// file      : xsd/cxx/xml/char-utf8.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_TRANSCODER
#define XSD_CXX_XML_TRANSCODER
#define XSD_CXX_XML_TRANSCODER_CHAR_UTF8

#include <string>
#include <cstddef> // std::size_t

#include <xercesc/util/XercesDefs.hpp> // XMLCh

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/exceptions.hxx>  // invalid_utf16_string

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      struct invalid_utf8_string {};

      // UTF-16 to/from UTF-8 transcoder.
      //
      template <typename C>
      struct char_utf8_transcoder
      {
        static std::basic_string<C>
        to (const XMLCh* s, std::size_t length);

        static XMLCh*
        from (const C* s, std::size_t length);

      private:
        static const unsigned char first_byte_mask_[5];
      };

      typedef char_utf8_transcoder<char> char_transcoder;
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/char-utf8.txx>

#else
#  ifndef XSD_CXX_XML_TRANSCODER_CHAR_UTF8
     //
     // If you get this error, it usually means that either you compiled
     // your schemas with different --char-encoding values or you included
     // some of the libxsd headers (e.g., xsd/cxx/xml/string.hxx) directly
     // without first including the correct xsd/cxx/xml/char-*.hxx header.
     //
#    error conflicting character encoding detected
#  endif
#endif // XSD_CXX_XML_TRANSCODER
