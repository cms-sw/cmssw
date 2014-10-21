// file      : xsd/cxx/tree/date-time-insertion.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // time_zone
      //
      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, const time_zone& z)
      {
        return s << z.zone_hours () << z.zone_minutes ();
      }

      // gday
      //
      template <typename S, typename C, typename B>
      ostream<S>&
      operator<< (ostream<S>& s, const gday<C, B>& x)
      {
        bool zp (x.zone_present ());

        s << x.day () << zp;

        if (zp)
        {
          const time_zone& z (x);
          s << z;
        }

        return s;
      }

      // gmonth
      //
      template <typename S, typename C, typename B>
      ostream<S>&
      operator<< (ostream<S>& s, const gmonth<C, B>& x)
      {
        bool zp (x.zone_present ());

        s << x.month () << zp;

        if (zp)
        {
          const time_zone& z (x);
          s << z;
        }

        return s;
      }

      // gyear
      //
      template <typename S, typename C, typename B>
      ostream<S>&
      operator<< (ostream<S>& s, const gyear<C, B>& x)
      {
        bool zp (x.zone_present ());

        s << x.year () << zp;

        if (zp)
        {
          const time_zone& z (x);
          s << z;
        }

        return s;
      }

      // gmonth_day
      //
      template <typename S, typename C, typename B>
      ostream<S>&
      operator<< (ostream<S>& s, const gmonth_day<C, B>& x)
      {
        bool zp (x.zone_present ());

        s << x.month () << x.day () << zp;

        if (zp)
        {
          const time_zone& z (x);
          s << z;
        }

        return s;
      }

      // gyear_month
      //
      template <typename S, typename C, typename B>
      ostream<S>&
      operator<< (ostream<S>& s, const gyear_month<C, B>& x)
      {
        bool zp (x.zone_present ());

        s << x.year () << x.month () << zp;

        if (zp)
        {
          const time_zone& z (x);
          s << z;
        }

        return s;
      }

      // date
      //
      template <typename S, typename C, typename B>
      ostream<S>&
      operator<< (ostream<S>& s, const date<C, B>& x)
      {
        bool zp (x.zone_present ());

        s << x.year () << x.month () << x.day () << zp;

        if (zp)
        {
          const time_zone& z (x);
          s << z;
        }

        return s;
      }

      // time
      //
      template <typename S, typename C, typename B>
      ostream<S>&
      operator<< (ostream<S>& s, const time<C, B>& x)
      {
        bool zp (x.zone_present ());

        s << x.hours () << x.minutes () << x.seconds () << zp;

        if (zp)
        {
          const time_zone& z (x);
          s << z;
        }

        return s;
      }

      // date_time
      //
      template <typename S, typename C, typename B>
      ostream<S>&
      operator<< (ostream<S>& s, const date_time<C, B>& x)
      {
        bool zp (x.zone_present ());

        s << x.year () << x.month () << x.day ()
          << x.hours () << x.minutes () << x.seconds () << zp;

        if (zp)
        {
          const time_zone& z (x);
          s << z;
        }

        return s;
      }

      // duration
      //
      template <typename S, typename C, typename B>
      ostream<S>&
      operator<< (ostream<S>& s, const duration<C, B>& x)
      {
        s << x.negative ()
          << x.years () << x.months () << x.days ()
          << x.hours () << x.minutes () << x.seconds ();

        return s;
      }
    }
  }
}
