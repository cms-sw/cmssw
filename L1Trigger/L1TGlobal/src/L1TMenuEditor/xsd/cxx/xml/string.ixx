// file      : xsd/cxx/xml/string.ixx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_STRING_IXX
#define XSD_CXX_XML_STRING_IXX

#include <xercesc/util/XMLString.hpp>

// If no transcoder has been included, use the default UTF-8.
//
#ifndef XSD_CXX_XML_TRANSCODER
#  include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/char-utf8.hxx>
#endif

// We sometimes need this functionality even if we are building for
// wchar_t.
//
namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      template <>
      inline std::basic_string<char>
      transcode<char> (const XMLCh* s)
      {
        if (s == 0 || *s == XMLCh (0))
          return std::basic_string<char> ();

#ifndef XSD_CXX_XML_TRANSCODER_CHAR_LCP
        return char_transcoder::to (s, xercesc::XMLString::stringLen (s));
#else
        return char_transcoder::to (s);
#endif
      }

      template <>
      inline std::basic_string<char>
      transcode<char> (const XMLCh* s, std::size_t len)
      {
        if (s == 0 || len == 0)
          return std::basic_string<char> ();

        return char_transcoder::to (s, len);
      }

      template <>
      inline XMLCh*
      transcode_to_xmlch (const char* s)
      {
#ifndef XSD_CXX_XML_TRANSCODER_CHAR_LCP
        return char_transcoder::from (s, std::char_traits<char>::length (s));
#else
        return char_transcoder::from (s);
#endif
      }

      template <>
      inline XMLCh*
      transcode_to_xmlch (const std::basic_string<char>& s)
      {
#ifndef XSD_CXX_XML_TRANSCODER_CHAR_LCP
        return char_transcoder::from (s.c_str (), s.length ());
#else
        return char_transcoder::from (s.c_str ());
#endif
      }
    }
  }
}

#endif // XSD_CXX_XML_STRING_IXX


#if defined(XSD_USE_CHAR) || !defined(XSD_USE_WCHAR)

#ifndef XSD_CXX_XML_STRING_IXX_CHAR
#define XSD_CXX_XML_STRING_IXX_CHAR

#endif // XSD_CXX_XML_STRING_IXX_CHAR
#endif // XSD_USE_CHAR


#if defined(XSD_USE_WCHAR) || !defined(XSD_USE_CHAR)

#ifndef XSD_CXX_XML_STRING_IXX_WCHAR
#define XSD_CXX_XML_STRING_IXX_WCHAR

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace bits
      {
        template <typename W, std::size_t S>
        struct wchar_transcoder;

        // Specialization for 2-byte wchar_t (resulting encoding is UTF-16).
        //
        template <typename W>
        struct wchar_transcoder<W, 2>
        {
          static std::basic_string<W>
          to (const XMLCh* s, std::size_t length);

          static XMLCh*
          from (const W* s, std::size_t length);
        };


        // Specialization for 4-byte wchar_t (resulting encoding is UCS-4).
        //
        template <typename W>
        struct wchar_transcoder<W, 4>
        {
          static std::basic_string<W>
          to (const XMLCh* s, std::size_t length);

          static XMLCh*
          from (const W* s, std::size_t length);
        };
      }

      template <>
      inline std::basic_string<wchar_t>
      transcode<wchar_t> (const XMLCh* s)
      {
        if (s == 0)
          return std::basic_string<wchar_t> ();

        return bits::wchar_transcoder<wchar_t, sizeof (wchar_t)>::to (
          s, xercesc::XMLString::stringLen (s));
      }

      template <>
      inline std::basic_string<wchar_t>
      transcode<wchar_t> (const XMLCh* s, std::size_t len)
      {
        if (s == 0 || len == 0)
          return std::basic_string<wchar_t> ();

        return bits::wchar_transcoder<wchar_t, sizeof (wchar_t)>::to (
          s, len);
      }

      template <>
      inline XMLCh*
      transcode_to_xmlch (const wchar_t* s)
      {
        return bits::wchar_transcoder<wchar_t, sizeof (wchar_t)>::from (
          s, std::char_traits<wchar_t>::length (s));
      }

      template <>
      inline XMLCh*
      transcode_to_xmlch (const std::basic_string<wchar_t>& s)
      {
        return bits::wchar_transcoder<wchar_t, sizeof (wchar_t)>::from (
          s.c_str (), s.length ());
      }
    }
  }
}

#endif // XSD_CXX_XML_STRING_IXX_WCHAR
#endif // XSD_USE_WCHAR
