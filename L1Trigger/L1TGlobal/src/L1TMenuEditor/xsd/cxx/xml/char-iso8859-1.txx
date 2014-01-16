// file      : xsd/cxx/xml/char-iso8859-1.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/auto-array.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      template <typename C>
      C char_iso8859_1_transcoder<C>::unrep_char_ = 0;

      template <typename C>
      std::basic_string<C> char_iso8859_1_transcoder<C>::
      to (const XMLCh* s, std::size_t len)
      {
        const XMLCh* end (s + len);

        // Find what the resulting buffer size will be.
        //
        std::size_t rl (0);
        unsigned int u (0); // Four byte UCS-4 char.

        bool valid (true);
        const XMLCh* p (s);

        for (; p < end; ++p)
        {
          if (*p >= 0xD800 && *p <= 0xDBFF)
          {
            // Make sure we have one more char and it has a valid
            // value for the second char in a surrogate pair.
            //
            if (++p == end || !((*p >= 0xDC00) && (*p <= 0xDFFF)))
            {
              valid = false;
              break;
            }
          }

          rl++;
        }

        if (!valid)
          throw invalid_utf16_string ();

        std::basic_string<C> r;
        r.reserve (rl + 1);
        r.resize (rl);
        C* rs (const_cast<C*> (r.c_str ()));
        std::size_t i (0);

        p = s;

        // Tight first loop for the common case.
        //
        for (; p < end && *p < 0x100; ++p)
          rs[i++] = C (*p);

        if (p < end && unrep_char_ == 0)
          throw iso8859_1_unrepresentable ();

        for (; p < end; ++p)
        {
          XMLCh x (*p);

          if ((x >= 0xD800) && (x <= 0xDBFF))
          {
            u = ((x - 0xD800) << 10) + (*++p - 0xDC00) + 0x10000;
          }
          else
            u = x;

          rs[i++] = u < 0x100 ? C (u) : unrep_char_;
        }

        return r;
      }

      template <typename C>
      XMLCh* char_iso8859_1_transcoder<C>::
      from (const C* s, std::size_t len)
      {
        const C* end (s + len);

        auto_array<XMLCh> r (new XMLCh[len + 1]);
        XMLCh* ir (r.get ());

        for (const C* p (s); p < end; ++p)
          *ir++ = static_cast<unsigned char> (*p);

        *ir = XMLCh (0);
        return r.release ();
      }
    }
  }
}
