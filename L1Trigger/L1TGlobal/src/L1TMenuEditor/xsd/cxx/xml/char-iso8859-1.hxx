// file      : xsd/cxx/xml/char-iso8859-1.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_TRANSCODER
#define XSD_CXX_XML_TRANSCODER
#define XSD_CXX_XML_TRANSCODER_CHAR_ISO8859_1

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
      struct iso8859_1_unrepresentable {};

      // UTF-16 to/from ISO-8859-1 transcoder.
      //
      template <typename C>
      struct char_iso8859_1_transcoder
      {
        static std::basic_string<C>
        to (const XMLCh* s, std::size_t length);

        static XMLCh*
        from (const C* s, std::size_t length);

        // Get/set a replacement for unrepresentable characters. If set to
        // 0 (the default value), throw iso8859_1_unrepresentable instead.
        //
        static C
        unrep_char ()
        {
          return unrep_char_;
        }

        static void
        unrep_char (C c)
        {
          unrep_char_ = c;
        }

      private:
        static C unrep_char_;
      };

      typedef char_iso8859_1_transcoder<char> char_transcoder;
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/char-iso8859-1.txx>

#else
#  ifndef XSD_CXX_XML_TRANSCODER_CHAR_ISO8859_1
     //
     // If you get this error, it usually means that either you compiled
     // your schemas with different --char-encoding values or you included
     // some of the libxsd headers (e.g., xsd/cxx/xml/string.hxx) directly
     // without first including the correct xsd/cxx/xml/char-*.hxx header.
     //
#    error conflicting character encoding detected
#  endif
#endif // XSD_CXX_XML_TRANSCODER
