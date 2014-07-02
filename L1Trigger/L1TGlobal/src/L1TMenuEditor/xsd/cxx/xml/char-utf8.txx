// file      : xsd/cxx/xml/char-utf8.txx
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
      const unsigned char char_utf8_transcoder<C>::first_byte_mask_[5] =
      {
        0x00, 0x00, 0xC0, 0xE0, 0xF0
      };

      template <typename C>
      std::basic_string<C> char_utf8_transcoder<C>::
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
          XMLCh x (*p);

          if (x < 0xD800 || x > 0xDBFF)
            u = x;
          else
          {
            // Make sure we have one more char and it has a valid
            // value for the second char in a surrogate pair.
            //
            if (++p == end || !((*p >= 0xDC00) && (*p <= 0xDFFF)))
            {
              valid = false;
              break;
            }

            u = ((x - 0xD800) << 10) + (*p - 0xDC00) + 0x10000;
          }

          if (u < 0x80)
            rl++;
          else if (u < 0x800)
            rl += 2;
          else if (u < 0x10000)
            rl += 3;
          else if (u < 0x110000)
            rl += 4;
          else
          {
            valid = false;
            break;
          }
        }

        if (!valid)
          throw invalid_utf16_string ();

        std::basic_string<C> r;
        r.reserve (rl + 1);
        r.resize (rl);
        C* rs (const_cast<C*> (r.c_str ()));

        std::size_t i (0);
        unsigned int count (0);

        p = s;

        // Tight first loop for the common case.
        //
        for (; p < end && *p < 0x80; ++p)
          rs[i++] = C (*p);

        for (; p < end; ++p)
        {
          XMLCh x (*p);

          if ((x >= 0xD800) && (x <= 0xDBFF))
          {
            u = ((x - 0xD800) << 10) + (*++p - 0xDC00) + 0x10000;
          }
          else
            u = x;

          if (u < 0x80)
            count = 1;
          else if (u < 0x800)
            count = 2;
          else if (u < 0x10000)
            count = 3;
          else if (u < 0x110000)
            count = 4;

          switch(count)
          {
          case 4:
            {
              rs[i + 3] = C ((u | 0x80UL) & 0xBFUL);
              u >>= 6;
            }
          case 3:
            {
              rs[i + 2] = C ((u | 0x80UL) & 0xBFUL);
              u >>= 6;
            }
          case 2:
            {
              rs[i + 1] = C ((u | 0x80UL) & 0xBFUL);
              u >>= 6;
            }
          case 1:
            {
              rs[i] = C (u | first_byte_mask_[count]);
            }
          }

          i += count;
        }

        return r;
      }

      template <typename C>
      XMLCh* char_utf8_transcoder<C>::
      from (const C* s, std::size_t len)
      {
        bool valid (true);
        const C* end (s + len);

        // Find what the resulting buffer size will be.
        //
        std::size_t rl (0);
        unsigned int count (0);

        for (const C* p (s); p < end; ++p)
        {
          unsigned char c (*p);

          if (c < 0x80)
          {
            // Fast path.
            //
            rl += 1;
            continue;
          }
          else if ((c >> 5) == 0x06)
            count = 2;
          else if ((c >> 4) == 0x0E)
            count = 3;
          else if ((c >> 3) == 0x1E)
            count = 4;
          else
          {
            valid = false;
            break;
          }

          p += count - 1; // One will be added in the for loop

          if (p + 1 > end)
          {
            valid = false;
            break;
          }

          // BMP is represented by up to 3 code points in UTF-8.
          //
          rl += count > 3 ? 2 : 1;
        }

        if (!valid)
          throw invalid_utf8_string ();

        auto_array<XMLCh> r (new XMLCh[rl + 1]);
        XMLCh* ir (r.get ());

        unsigned int u (0); // Four byte UCS-4 char.

        for (const C* p (s); p < end; ++p)
        {
          unsigned char c (*p);

          if (c < 0x80)
          {
            // Fast path.
            //
            *ir++ = static_cast<XMLCh> (c);
            continue;
          }
          else if ((c >> 5) == 0x06)
          {
            // UTF-8:   110yyyyy 10zzzzzz
            // Unicode: 00000yyy yyzzzzzz
            //
            u = (c & 0x1F) << 6;

            c = *++p;
            if ((c >> 6) != 2)
            {
              valid = false;
              break;
            }
            u |= c & 0x3F;
          }
          else if ((c >> 4) == 0x0E)
          {
            // UTF-8:   1110xxxx 10yyyyyy 10zzzzzz
            // Unicode: xxxxyyyy yyzzzzzz
            //
            u = (c & 0x0F) << 6;

            c = *++p;
            if ((c >> 6) != 2)
            {
              valid = false;
              break;
            }
            u = (u | (c & 0x3F)) << 6;

            c = *++p;
            if ((c >> 6) != 2)
            {
              valid = false;
              break;
            }
            u |= c & 0x3F;
          }
          else if ((c >> 3) == 0x1E)
          {
            // UTF-8:   000wwwxx xxxxyyyy yyzzzzzz
            // Unicode: 11110www 10xxxxxx 10yyyyyy 10zzzzzz
            //
            u = (c & 0x07) << 6;

            c = *++p;
            if ((c >> 6) != 2)
            {
              valid = false;
              break;
            }
            u = (u | (c & 0x3F)) << 6;

            c = *++p;
            if ((c >> 6) != 2)
            {
              valid = false;
              break;
            }
            u = (u | (c & 0x3F)) << 6;

            c = *++p;
            if ((c >> 6) != 2)
            {
              valid = false;
              break;
            }
            u |= c & 0x3F;
          }

          if (u & 0xFFFF0000)
          {
            // Surrogate pair.
            //
            *ir++ = static_cast<XMLCh> (((u - 0x10000) >> 10) + 0xD800);
            *ir++ = static_cast<XMLCh> ((u & 0x3FF) + 0xDC00);
          }
          else
            *ir++ = static_cast<XMLCh> (u);
        }

        if (!valid)
          throw invalid_utf8_string ();

        *ir = XMLCh (0);

        return r.release ();
      }
    }
  }
}
