// file      : xsd/cxx/xml/string.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_STRING_TXX
#define XSD_CXX_XML_STRING_TXX


#endif // XSD_CXX_XML_STRING_TXX

#if defined(XSD_USE_WCHAR) || !defined(XSD_USE_CHAR)

#ifndef XSD_CXX_XML_STRING_TXX_WCHAR
#define XSD_CXX_XML_STRING_TXX_WCHAR

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/exceptions.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace bits
      {
        // wchar_transcoder (specialization for 2-byte wchar_t)
        //
        template <typename W>
        std::basic_string<W> wchar_transcoder<W, 2>::
        to (const XMLCh* s, std::size_t length)
        {
          std::basic_string<W> r;
          r.reserve (length + 1);
          r.resize (length);
          W* rs (const_cast<W*> (r.c_str ()));

          for (std::size_t i (0); i < length; ++s, ++i)
          {
            rs[i] = *s;
          }

          return r;
        }

        template <typename W>
        XMLCh* wchar_transcoder<W, 2>::
        from (const W* s, std::size_t length)
        {
          auto_array<XMLCh> r (new XMLCh[length + 1]);
          XMLCh* ir (r.get ());

          for (std::size_t i (0); i < length; ++ir, ++i)
          {
            *ir = static_cast<XMLCh> (s[i]);
          }

          *ir = XMLCh (0);

          return r.release ();
        }


        // wchar_transcoder (specialization for 4-byte wchar_t)
        //
        template <typename W>
        std::basic_string<W> wchar_transcoder<W, 4>::
        to (const XMLCh* s, std::size_t length)
        {
          const XMLCh* end (s + length);

          // Find what the resulting buffer size will be.
          //
          std::size_t rl (0);

          for (const XMLCh* p (s); p < end; ++p)
          {
            rl++;

            if ((*p >= 0xD800) && (*p <= 0xDBFF))
            {
              // Make sure we have one more char and it has a valid
              // value for the second char in a surrogate pair.
              //
              if (++p == end || !((*p >= 0xDC00) && (*p <= 0xDFFF)))
                throw invalid_utf16_string ();
            }
          }

          std::basic_string<W> r;
          r.reserve (rl + 1);
          r.resize (rl);
          W* rs (const_cast<W*> (r.c_str ()));

          std::size_t i (0);

          for (const XMLCh* p (s); p < end; ++p)
          {
            XMLCh x (*p);

            if (x < 0xD800 || x > 0xDBFF)
              rs[i++] = W (x);
            else
              rs[i++] = ((x - 0xD800) << 10) + (*++p - 0xDC00) + 0x10000;
          }

          return r;
        }

        template <typename W>
        XMLCh* wchar_transcoder<W, 4>::
        from (const W* s, std::size_t length)
        {
          // Find what the resulting buffer size will be.
          //
          std::size_t rl (0);

          for (const W* p (s); p < s + length; ++p)
          {
            rl += (*p & 0xFFFF0000) ? 2 : 1;
          }

          auto_array<XMLCh> r (new XMLCh[rl + 1]);
          XMLCh* ir (r.get ());

          for (const W* p (s); p < s + length; ++p)
          {
            W w (*p);

            if (w & 0xFFFF0000)
            {
              // Surrogate pair.
              //
              *ir++ = static_cast<XMLCh> (((w - 0x10000) >> 10) + 0xD800);
              *ir++ = static_cast<XMLCh> ((w & 0x3FF) + 0xDC00);
            }
            else
              *ir++ = static_cast<XMLCh> (w);
          }

          *ir = XMLCh (0);

          return r.release ();
        }
      }
    }
  }
}

#endif // XSD_CXX_XML_STRING_TXX_WCHAR
#endif // XSD_USE_WCHAR
