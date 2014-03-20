// file      : xsd/cxx/tree/date-time-ostream.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <ostream>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // time_zone
      //
      template <typename C>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const time_zone& z)
      {
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

          C f (os.fill (C ('0')));

          os.width (2);
          os << h << C (':');
          os.width (2);
          os << m;

          os.fill (f);
        }

        return os;
      }

      // gday
      //
      template <typename C, typename B>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const gday<C, B>& x)
      {
        C f (os.fill (C ('0')));
        os.width (2);
        os << x.day ();
        os.fill (f);

        if (x.zone_present ())
        {
          const time_zone& z (x);
          os << z;
        }

        return os;
      }

      // gmonth
      //
      template <typename C, typename B>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const gmonth<C, B>& x)
      {
        C f (os.fill (C ('0')));
        os.width (2);
        os << x.month ();
        os.fill (f);

        if (x.zone_present ())
        {
          const time_zone& z (x);
          os << z;
        }

        return os;
      }

      // gyear
      //
      template <typename C, typename B>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const gyear<C, B>& x)
      {
        C f (os.fill (C ('0')));
        os.width (4);
        os << x.year ();
        os.fill (f);

        if (x.zone_present ())
        {
          const time_zone& z (x);
          os << z;
        }

        return os;
      }

      // gmonth_day
      //
      template <typename C, typename B>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const gmonth_day<C, B>& x)
      {
        C f (os.fill (C ('0')));

        os.width (2);
        os << x.month () << C ('-');

        os.width (2);
        os << x.day ();

        os.fill (f);

        if (x.zone_present ())
        {
          const time_zone& z (x);
          os << z;
        }

        return os;
      }


      // gyear_month
      //
      template <typename C, typename B>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const gyear_month<C, B>& x)
      {
        C f (os.fill (C ('0')));

        os.width (4);
        os << x.year () << C ('-');

        os.width (2);
        os << x.month ();

        os.fill (f);

        if (x.zone_present ())
        {
          const time_zone& z (x);
          os << z;
        }

        return os;
      }

      // date
      //
      template <typename C, typename B>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const date<C, B>& x)
      {
        C f (os.fill (C ('0')));

        os.width (4);
        os << x.year () << C ('-');

        os.width (2);
        os << x.month () << C ('-');

        os.width (2);
        os << x.day ();

        os.fill (f);

        if (x.zone_present ())
        {
          const time_zone& z (x);
          os << z;
        }

        return os;
      }

      // time
      //
      template <typename C, typename B>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const time<C, B>& x)
      {
        C f (os.fill (C ('0')));

        os.width (2);
        os << x.hours () << C (':');

        os.width (2);
        os << x.minutes () << C (':');

        os.width (9);
        std::ios_base::fmtflags ff (
          os.setf (std::ios::fixed, std::ios::floatfield));
        os << x.seconds ();
        os.setf (ff, std::ios::floatfield);

        os.fill (f);

        if (x.zone_present ())
        {
          const time_zone& z (x);
          os << z;
        }

        return os;
      }

      // date_time
      //
      template <typename C, typename B>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const date_time<C, B>& x)
      {
        C f (os.fill (C ('0')));

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

        os.width (9);
        std::ios_base::fmtflags ff (
          os.setf (std::ios::fixed, std::ios::floatfield));
        os << x.seconds ();
        os.setf (ff, std::ios::floatfield);

        os.fill (f);

        if (x.zone_present ())
        {
          const time_zone& z (x);
          os << z;
        }

        return os;
      }

      // duration
      //
      template <typename C, typename B>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const duration<C, B>& x)
      {
        if (x.negative ())
          os << C ('-');

        os << C ('P');

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

        if (x.months () != 0)
        {
          os << x.months () << C ('M');
        }

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

        if (x.hours () != 0)
        {
          os << x.hours () << C ('H');
        }

        if (x.minutes () != 0)
        {
          os << x.minutes () << C ('M');
        }

        if (x.seconds () > 0.0)
        {
          std::ios_base::fmtflags ff (
            os.setf (std::ios::fixed, std::ios::floatfield));
          os << x.seconds () << C ('S');
          os.setf (ff, std::ios::floatfield);
        }

        return os;
      }
    }
  }
}
