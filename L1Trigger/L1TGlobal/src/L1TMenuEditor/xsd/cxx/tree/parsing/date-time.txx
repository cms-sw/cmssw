// file      : xsd/cxx/tree/parsing/date-time.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/zc-istream.hxx>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.hxx> // xml::transcode

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/text.hxx>  // text_content

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // time_zone
      //
      template <typename C>
      void time_zone::
      zone_parse (const C* s, std::size_t n)
      {
        // time_zone := Z|(+|-)HH:MM
        //
        if (n == 0)
        {
          return;
        }
        else if (s[0] == C ('Z'))
        {
          hours_ = 0;
          minutes_ = 0;
          present_ = true;
        }
        else if (n == 6)
        {
          // Parse hours.
          //
          hours_ = 10 * (s[1] - C ('0')) + (s[2] - C ('0'));

          // Parse minutes.
          //
          minutes_ = 10 * (s[4] - C ('0')) + (s[5] - C ('0'));

          if (s[0] == C ('-'))
          {
            hours_ = -hours_;
            minutes_ = -minutes_;
          }
          present_ = true;
        }
      }

      // gday
      //
      template <typename C, typename B>
      gday<C, B>::
      gday (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        parse (text_content<C> (e));
      }

      template <typename C, typename B>
      gday<C, B>::
      gday (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        parse (xml::transcode<C> (a.getValue ()));
      }

      template <typename C, typename B>
      gday<C, B>::
      gday (const std::basic_string<C>& s,
            const xercesc::DOMElement* e,
            flags f,
            container* c)
          : B (s, e, f, c)
      {
        parse (s);
      }

      template <typename C, typename B>
      void gday<C, B>::
      parse (const std::basic_string<C>& str)
      {
        typedef typename ro_string<C>::size_type size_type;

        ro_string<C> tmp (str);
        size_type n (trim (tmp));
        const C* s (tmp.data ());

        // gday := ---DD[Z|(+|-)HH:MM]
        //
        if (n >= 5)
        {
          day_ = 10 * (s[3] - C ('0')) + (s[4] - C ('0'));

          if (n > 5)
            zone_parse (s + 5, n - 5);
        }
      }

      // gmonth
      //
      template <typename C, typename B>
      gmonth<C, B>::
      gmonth (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        parse (text_content<C> (e));
      }

      template <typename C, typename B>
      gmonth<C, B>::
      gmonth (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        parse (xml::transcode<C> (a.getValue ()));
      }

      template <typename C, typename B>
      gmonth<C, B>::
      gmonth (const std::basic_string<C>& s,
              const xercesc::DOMElement* e,
              flags f,
              container* c)
          : B (s, e, f, c)
      {
        parse (s);
      }

      template <typename C, typename B>
      void gmonth<C, B>::
      parse (const std::basic_string<C>& str)
      {
        typedef typename ro_string<C>::size_type size_type;

        ro_string<C> tmp (str);
        size_type n (trim (tmp));
        const C* s (tmp.data ());

        // gmonth := --MM[Z|(+|-)HH:MM]
        //
        if (n >= 4)
        {
          month_ = 10 * (s[2] - C ('0')) + (s[3] - C ('0'));

          if (n > 4)
            zone_parse (s + 4, n - 4);
        }
      }

      // gyear
      //
      template <typename C, typename B>
      gyear<C, B>::
      gyear (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        parse (text_content<C> (e));
      }

      template <typename C, typename B>
      gyear<C, B>::
      gyear (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        parse (xml::transcode<C> (a.getValue ()));
      }

      template <typename C, typename B>
      gyear<C, B>::
      gyear (const std::basic_string<C>& s,
             const xercesc::DOMElement* e,
             flags f,
             container* c)
          : B (s, e, f, c)
      {
        parse (s);
      }

      template <typename C, typename B>
      void gyear<C, B>::
      parse (const std::basic_string<C>& str)
      {
        typedef typename ro_string<C>::size_type size_type;

        ro_string<C> tmp (str);
        size_type n (trim (tmp));
        const C* s (tmp.data ());

        // gyear := [-]CCYY[N]*[Z|(+|-)HH:MM]
        //
        if (n >= 4)
        {
          // Find the end of the year token.
          //
          size_type pos (4);
          for (; pos < n; ++pos)
          {
            C c (s[pos]);

            if (c == C ('Z') || c == C ('+') || c == C ('-'))
              break;
          }

          ro_string<C> year_fragment (s, pos);
          zc_istream<C> is (year_fragment);
          is >> year_;

          if (pos < n)
            zone_parse (s + pos, n - pos);
        }
      }

      // gmonth_day
      //
      template <typename C, typename B>
      gmonth_day<C, B>::
      gmonth_day (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        parse (text_content<C> (e));
      }

      template <typename C, typename B>
      gmonth_day<C, B>::
      gmonth_day (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        parse (xml::transcode<C> (a.getValue ()));
      }

      template <typename C, typename B>
      gmonth_day<C, B>::
      gmonth_day (const std::basic_string<C>& s,
                  const xercesc::DOMElement* e,
                  flags f,
                  container* c)
          : B (s, e, f, c)
      {
        parse (s);
      }

      template <typename C, typename B>
      void gmonth_day<C, B>::
      parse (const std::basic_string<C>& str)
      {
        typedef typename ro_string<C>::size_type size_type;

        ro_string<C> tmp (str);
        size_type n (trim (tmp));
        const C* s (tmp.data ());

        // gmonth_day := --MM-DD[Z|(+|-)HH:MM]
        //
        if (n >= 7)
        {
          month_ = 10 * (s[2] - C ('0')) + (s[3] - C ('0'));
          day_ = 10 * (s[5] - C ('0')) + (s[6] - C ('0'));

          if (n > 7)
            zone_parse (s + 7, n - 7);
        }
      }

      // gyear_month
      //
      template <typename C, typename B>
      gyear_month<C, B>::
      gyear_month (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        parse (text_content<C> (e));
      }

      template <typename C, typename B>
      gyear_month<C, B>::
      gyear_month (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        parse (xml::transcode<C> (a.getValue ()));
      }

      template <typename C, typename B>
      gyear_month<C, B>::
      gyear_month (const std::basic_string<C>& s,
                   const xercesc::DOMElement* e,
                   flags f,
                   container* c)
          : B (s, e, f, c)
      {
        parse (s);
      }

      template <typename C, typename B>
      void gyear_month<C, B>::
      parse (const std::basic_string<C>& str)
      {
        typedef typename ro_string<C>::size_type size_type;

        ro_string<C> tmp (str);
        size_type n (trim (tmp));
        const C* s (tmp.data ());

        // gyear_month := [-]CCYY[N]*-MM[Z|(+|-)HH:MM]
        //

        if (n >= 7)
        {
          // Find the end of the year token.
          //
          size_type pos (tmp.find (C ('-'), 4));

          if (pos != ro_string<C>::npos && (n - pos - 1) >= 2)
          {
            ro_string<C> year_fragment (s, pos);
            zc_istream<C> is (year_fragment);
            is >> year_;

            month_ = 10 * (s[pos + 1] - C ('0')) + (s[pos + 2] - C ('0'));

            pos += 3;

            if (pos < n)
              zone_parse (s + pos, n - pos);
          }
        }
      }

      // date
      //
      template <typename C, typename B>
      date<C, B>::
      date (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        parse (text_content<C> (e));
      }

      template <typename C, typename B>
      date<C, B>::
      date (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        parse (xml::transcode<C> (a.getValue ()));
      }

      template <typename C, typename B>
      date<C, B>::
      date (const std::basic_string<C>& s,
            const xercesc::DOMElement* e,
            flags f,
            container* c)
          : B (s, e, f, c)
      {
        parse (s);
      }

      template <typename C, typename B>
      void date<C, B>::
      parse (const std::basic_string<C>& str)
      {
        typedef typename ro_string<C>::size_type size_type;

        ro_string<C> tmp (str);
        size_type n (trim (tmp));
        const C* s (tmp.data ());

        // date := [-]CCYY[N]*-MM-DD[Z|(+|-)HH:MM]
        //

        if (n >= 10)
        {
          // Find the end of the year token.
          //
          size_type pos (tmp.find (C ('-'), 4));

          if (pos != ro_string<C>::npos && (n - pos - 1) >= 5)
          {
            ro_string<C> year_fragment (s, pos);
            zc_istream<C> is (year_fragment);
            is >> year_;

            month_ = 10 * (s[pos + 1] - C ('0')) + (s[pos + 2] - C ('0'));
            day_ = 10 * (s[pos + 4] - C ('0')) + (s[pos + 5] - C ('0'));

            pos += 6;

            if (pos < n)
              zone_parse (s + pos, n - pos);
          }
        }
      }

      // time
      //
      template <typename C, typename B>
      time<C, B>::
      time (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        parse (text_content<C> (e));
      }

      template <typename C, typename B>
      time<C, B>::
      time (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        parse (xml::transcode<C> (a.getValue ()));
      }

      template <typename C, typename B>
      time<C, B>::
      time (const std::basic_string<C>& s,
            const xercesc::DOMElement* e,
            flags f,
            container* c)
          : B (s, e, f, c)
      {
        parse (s);
      }

      template <typename C, typename B>
      void time<C, B>::
      parse (const std::basic_string<C>& str)
      {
        typedef typename ro_string<C>::size_type size_type;

        ro_string<C> tmp (str);
        size_type n (trim (tmp));
        const C* s (tmp.data ());

        // time := HH:MM:SS[.S+][Z|(+|-)HH:MM]
        //

        if (n >= 8)
        {
          hours_ = 10 * (s[0] - '0') + (s[1] - '0');
          minutes_ = 10 * (s[3] - '0') + (s[4] - '0');

          // Find the end of the seconds fragment.
          //
          size_type pos (8);
          for (; pos < n; ++pos)
          {
            C c (s[pos]);

            if (c == C ('Z') || c == C ('+') || c == C ('-'))
              break;
          }

          ro_string<C> seconds_fragment (s + 6, pos - 6);
          zc_istream<C> is (seconds_fragment);
          is >> seconds_;

          if (pos < n)
            zone_parse (s + pos, n - pos);
        }
      }

      // date_time
      //
      template <typename C, typename B>
      date_time<C, B>::
      date_time (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        parse (text_content<C> (e));
      }

      template <typename C, typename B>
      date_time<C, B>::
      date_time (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        parse (xml::transcode<C> (a.getValue ()));
      }

      template <typename C, typename B>
      date_time<C, B>::
      date_time (const std::basic_string<C>& s,
                 const xercesc::DOMElement* e,
                 flags f,
                 container* c)
          : B (s, e, f, c)
      {
        parse (s);
      }

      template <typename C, typename B>
      void date_time<C, B>::
      parse (const std::basic_string<C>& str)
      {
        typedef typename ro_string<C>::size_type size_type;

        ro_string<C> tmp (str);
        size_type n (trim (tmp));
        const C* s (tmp.data ());

        // date_time := [-]CCYY[N]*-MM-DDTHH:MM:SS[.S+][Z|(+|-)HH:MM]
        //

        if (n >= 19)
        {
          // Find the end of the year token.
          //
          size_type pos (tmp.find (C ('-'), 4));

          if (pos != ro_string<C>::npos && (n - pos - 1) >= 14)
          {
            ro_string<C> year_fragment (s, pos);
            zc_istream<C> yis (year_fragment);
            yis >> year_;

            month_ = 10 * (s[pos + 1] - C ('0')) + (s[pos + 2] - C ('0'));
            pos += 3;

            day_ = 10 * (s[pos + 1] - C ('0')) + (s[pos + 2] - C ('0'));
            pos += 3;

            hours_ = 10 * (s[pos + 1] - C ('0')) + (s[pos + 2] - C ('0'));
            pos += 3;

            minutes_ = 10 * (s[pos + 1] - C ('0')) + (s[pos + 2] - C ('0'));
            pos += 4; // Point to the first S.

            // Find the end of the seconds fragment.
            //
            size_type sec_end (pos + 2);
            for (; sec_end < n; ++sec_end)
            {
              C c (s[sec_end]);

              if (c == C ('Z') || c == C ('+') || c == C ('-'))
                break;
            }

            ro_string<C> seconds_fragment (s + pos, sec_end - pos);
            zc_istream<C> sis (seconds_fragment);
            sis >> seconds_;

            if (sec_end < n)
              zone_parse (s + sec_end, n - sec_end);
          }
        }
      }

      // duration
      //
      template <typename C, typename B>
      duration<C, B>::
      duration (const xercesc::DOMElement& e, flags f, container* c)
          : B (e, f, c)
      {
        parse (text_content<C> (e));
      }

      template <typename C, typename B>
      duration<C, B>::
      duration (const xercesc::DOMAttr& a, flags f, container* c)
          : B (a, f, c)
      {
        parse (xml::transcode<C> (a.getValue ()));
      }

      template <typename C, typename B>
      duration<C, B>::
      duration (const std::basic_string<C>& s,
                const xercesc::DOMElement* e,
                flags f,
                container* c)
          : B (s, e, f, c)
      {
        parse (s);
      }

      namespace bits
      {
        template <typename C>
        inline typename ro_string<C>::size_type
        duration_delim (const C* s,
                        typename ro_string<C>::size_type pos,
                        typename ro_string<C>::size_type size)
        {
          const C* p (s + pos);
          for (; p < (s + size); ++p)
          {
            if (*p == C ('Y') || *p == C ('D') || *p == C ('M') ||
                *p == C ('H') || *p == C ('M') || *p == C ('S') ||
                *p == C ('T'))
              break;
          }

          return p - s;
        }
      }

      template <typename C, typename B>
      void duration<C, B>::
      parse (const std::basic_string<C>& str)
      {
        typedef typename ro_string<C>::size_type size_type;

        ro_string<C> tmp (str);
        size_type n (trim (tmp));
        const C* s (tmp.data ());

        // Set all the fields since some of them may not be specified.
        //
        years_ = months_ = days_ = hours_ = minutes_ = 0;
        seconds_ = 0.0;

        // duration := [-]P[nY][nM][nD][TnHnMn[.n+]S]
        //
        if (n >= 3)
        {
          size_type pos (0);

          if (s[0] == C ('-'))
          {
            negative_ = true;
            pos++;
          }
          else
            negative_ = false;

          pos++; // Skip 'P'.

          size_type del (bits::duration_delim (s, pos, n));

          if (del != n && s[del] == C ('Y'))
          {
            ro_string<C> fragment (s + pos, del - pos);
            zc_istream<C> is (fragment);
            is >> years_;

            pos = del + 1;
            del = bits::duration_delim (s, pos, n);
          }

          if (del != n && s[del] == C ('M'))
          {
            ro_string<C> fragment (s + pos, del - pos);
            zc_istream<C> is (fragment);
            is >> months_;

            pos = del + 1;
            del = bits::duration_delim (s, pos, n);
          }

          if (del != n && s[del] == C ('D'))
          {
            ro_string<C> fragment (s + pos, del - pos);
            zc_istream<C> is (fragment);
            is >> days_;

            pos = del + 1;
            del = bits::duration_delim (s, pos, n);
          }

          if (del != n && s[del] == C ('T'))
          {
            pos = del + 1;
            del = bits::duration_delim (s, pos, n);

            if (del != n && s[del] == C ('H'))
            {
              ro_string<C> fragment (s + pos, del - pos);
              zc_istream<C> is (fragment);
              is >> hours_;

              pos = del + 1;
              del = bits::duration_delim (s, pos, n);
            }

            if (del != n && s[del] == C ('M'))
            {
              ro_string<C> fragment (s + pos, del - pos);
              zc_istream<C> is (fragment);
              is >> minutes_;

              pos = del + 1;
              del = bits::duration_delim (s, pos, n);
            }

            if (del != n && s[del] == C ('S'))
            {
              ro_string<C> fragment (s + pos, del - pos);
              zc_istream<C> is (fragment);
              is >> seconds_;
            }
          }
        }
      }
    }
  }
}
