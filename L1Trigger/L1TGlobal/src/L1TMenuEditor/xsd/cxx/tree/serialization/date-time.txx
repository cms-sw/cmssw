// file      : xsd/cxx/tree/serialization/date-time.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <locale>
#include <string>
#include <ostream>
#include <sstream>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/bits/literals.hxx> // bits::{gday_prefix,gmonth_prefix}

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // time_zone
      //
      namespace bits
      {
        // Assumes the fill character is set to '0'.
        //
        template <typename C>
        void
        zone_insert (std::basic_ostream<C>& os, const time_zone& z)
        {
          // time-zone := Z|(+|-)HH:MM
          //
          short h = z.zone_hours ();
          short m = z.zone_minutes ();

          if (h == 0 && m == 0)
          {
            os << C ('Z');
          }
          else
          {
            if (h < 0 || m < 0)
            {
              h = -h;
              m = -m;
              os << C ('-');
            }
            else
              os << C ('+');

            if (h >= 0 && h <= 14 && m >= 0 && m <= 59)
            {
              os.width (2);
              os << h << C (':');
              os.width (2);
              os << m;
            }
          }
        }
      }

      // gday
      //
      namespace bits
      {
        template <typename C, typename B>
        void
        insert (std::basic_ostream<C>& os, const tree::gday<C, B>& x)
        {
          if (x.day () < 32)
          {
            // Save some time and space by not restoring the fill character
            // since it is the same in case of a list.
            //
            os.fill (C ('0'));
            os << bits::gday_prefix<C> ();
            os.width (2);
            os << x.day ();

            if (x.zone_present ())
              zone_insert (os, x);
          }
        }
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const gday<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        e << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const gday<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        a << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const gday<C, B>& x)
      {
        bits::insert (ls.os_, x);
      }

      // gmonth
      //
      namespace bits
      {
        template <typename C, typename B>
        void
        insert (std::basic_ostream<C>& os, const tree::gmonth<C, B>& x)
        {
          if (x.month () < 13)
          {
            os.fill (C ('0'));
            os << bits::gmonth_prefix<C> ();
            os.width (2);
            os << x.month ();

            if (x.zone_present ())
              zone_insert (os, x);
          }
        }
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const gmonth<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        e << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const gmonth<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        a << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const gmonth<C, B>& x)
      {
        bits::insert (ls.os_, x);
      }

      // gyear
      //
      namespace bits
      {
        template <typename C, typename B>
        void
        insert (std::basic_ostream<C>& os, const tree::gyear<C, B>& x)
        {
          os.fill (C ('0'));
          os.width (4);
          os << x.year ();

          if (x.zone_present ())
            zone_insert (os, x);
        }
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const gyear<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        e << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const gyear<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        a << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const gyear<C, B>& x)
      {
        bits::insert (ls.os_, x);
      }

      // gmonth_day
      //
      namespace bits
      {
        template <typename C, typename B>
        void
        insert (std::basic_ostream<C>& os, const tree::gmonth_day<C, B>& x)
        {
          if (x.month () < 13 && x.day () < 32)
          {
            os.fill (C ('0'));
            os << bits::gmonth_prefix<C> ();
            os.width (2);
            os << x.month () << C ('-');
            os.width (2);
            os << x.day ();

            if (x.zone_present ())
              zone_insert (os, x);
          }
        }
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const gmonth_day<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        e << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const gmonth_day<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        a << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const gmonth_day<C, B>& x)
      {
        bits::insert (ls.os_, x);
      }

      // gyear_month
      //
      namespace bits
      {
        template <typename C, typename B>
        void
        insert (std::basic_ostream<C>& os, const tree::gyear_month<C, B>& x)
        {
          if (x.month () < 13)
          {
            os.fill (C ('0'));
            os.width (4);
            os << x.year () << C ('-');
            os.width (2);
            os << x.month ();

            if (x.zone_present ())
              zone_insert (os, x);
          }
        }
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const gyear_month<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        e << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const gyear_month<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        a << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const gyear_month<C, B>& x)
      {
        bits::insert (ls.os_, x);
      }

      // date
      //
      namespace bits
      {
        template <typename C, typename B>
        void
        insert (std::basic_ostream<C>& os, const tree::date<C, B>& x)
        {
          if (x.month () < 13 && x.day () < 32)
          {
            os.fill (C ('0'));
            os.width (4);
            os << x.year () << C ('-');
            os.width (2);
            os << x.month () << C ('-');
            os.width (2);
            os << x.day ();

            if (x.zone_present ())
              zone_insert (os, x);
          }
        }
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const date<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        e << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const date<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        a << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const date<C, B>& x)
      {
        bits::insert (ls.os_, x);
      }

      // time
      //
      namespace bits
      {
        template <typename C, typename B>
        void
        insert (std::basic_ostream<C>& os, const tree::time<C, B>& x)
        {
          if (x.hours () <= 24 &&
              x.minutes () <= 59 &&
              x.seconds () >= 0.0 &&
              x.seconds () < 60.0)
          {
            os.fill (C ('0'));
            os.width (2);
            os << x.hours () << C (':');

            os.width (2);
            os << x.minutes () << C (':');

            std::basic_ostringstream<C> ostr;
            ostr.imbue (std::locale::classic ());
            ostr.width (9);
            ostr.fill (C ('0'));
            ostr << std::fixed << x.seconds ();

            std::basic_string<C> s (ostr.str ());

            // Remove the trailing zeros and the decimal point if necessary.
            //
            typedef typename std::basic_string<C>::size_type size_type;

            size_type size (s.size ()), n (size);

            for (; n > 0 && s[n - 1] == C ('0'); --n)/*noop*/;

            if (n > 0 && s[n - 1] == C ('.'))
              --n;

            if (n != size)
              s.resize (n);

            os << s;

            if (x.zone_present ())
              zone_insert (os, x);
          }
        }
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const time<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        e << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const time<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        a << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const time<C, B>& x)
      {
        bits::insert (ls.os_, x);
      }

      // date_time
      //
      namespace bits
      {
        template <typename C, typename B>
        void
        insert (std::basic_ostream<C>& os, const tree::date_time<C, B>& x)
        {
          if (x.month () <= 12 &&
              x.day () <= 31 &&
              x.hours () <= 24 &&
              x.minutes () <= 59 &&
              x.seconds () >= 0.0 &&
              x.seconds () < 60.0)
          {
            os.fill (C ('0'));
            os.width (4);
            os << x.year () << C ('-');

            os.width (2);
            os << x.month () << C ('-');

            os.width (2);
            os << x.day () << C ('T');

            os.width (2);
            os << x.hours () << C (':');

            os.width (2);
            os << x.minutes () << C (':');

            std::basic_ostringstream<C> ostr;
            ostr.imbue (std::locale::classic ());
            ostr.width (9);
            ostr.fill (C ('0'));
            ostr << std::fixed << x.seconds ();

            std::basic_string<C> s (ostr.str ());

            // Remove the trailing zeros and the decimal point if necessary.
            //
            typedef typename std::basic_string<C>::size_type size_type;

            size_type size (s.size ()), n (size);

            for (; n > 0 && s[n - 1] == C ('0'); --n)/*noop*/;

            if (n > 0 && s[n - 1] == C ('.'))
              --n;

            if (n != size)
              s.resize (n);

            os << s;

            if (x.zone_present ())
              zone_insert (os, x);
          }
        }
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const date_time<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        e << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const date_time<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        a << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const date_time<C, B>& x)
      {
        bits::insert (ls.os_, x);
      }

      // duration
      //
      namespace bits
      {
        template <typename C, typename B>
        void
        insert (std::basic_ostream<C>& os, const tree::duration<C, B>& x)
        {
          if (x.negative ())
            os << C ('-');

          os << C ('P');

          // years
          //
          // In case it is 0-duration, use the years field to handle
          // this case.
          //
          if (x.years () != 0 ||
              (x.months () == 0 &&
               x.days () == 0 &&
               x.hours () == 0 &&
               x.minutes () == 0 &&
               x.seconds () == 0.0))
          {
            os << x.years () << C ('Y');
          }

          // months
          //
          if (x.months () != 0)
          {
            os << x.months () << C ('M');
          }

          // days
          //
          if (x.days () != 0)
          {
            os << x.days () << C ('D');
          }

          // Figure out if we need the 'T' delimiter.
          //
          if (x.hours () != 0 ||
              x.minutes () != 0 ||
              x.seconds () != 0.0)
            os << C ('T');

          // hours
          //
          if (x.hours () != 0)
          {
            os << x.hours () << C ('H');
          }

          // minutes
          //
          if (x.minutes () != 0)
          {
            os << x.minutes () << C ('M');
          }

          // seconds
          //
          if (x.seconds () > 0.0)
          {
            std::basic_ostringstream<C> ostr;
            ostr.imbue (std::locale::classic ());
            ostr << std::fixed << x.seconds ();

            std::basic_string<C> s (ostr.str ());

            // Remove the trailing zeros and the decimal point if necessary.
            //
            typedef typename std::basic_string<C>::size_type size_type;

            size_type size (s.size ()), n (size);

            for (; n > 0 && s[n - 1] == C ('0'); --n)/*noop*/;

            if (n > 0 && s[n - 1] == C ('.'))
              --n;

            if (n != size)
              s.resize (n);

            os << s << C ('S');
          }
        }
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMElement& e, const duration<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        e << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (xercesc::DOMAttr& a, const duration<C, B>& x)
      {
        std::basic_ostringstream<C> os;
        bits::insert (os, x);
        a << os.str ();
      }

      template <typename C, typename B>
      inline void
      operator<< (list_stream<C>& ls, const duration<C, B>& x)
      {
        bits::insert (ls.os_, x);
      }
    }
  }
}
