// file      : xsd/cxx/tree/date-time-extraction.txx
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
      inline void time_zone::
      zone_extract (istream<S>& s)
      {
        s >> hours_ >> minutes_;
        present_ = true;
      }

      // gday
      //
      template <typename C, typename B>
      template <typename S>
      gday<C, B>::
      gday (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        bool zp;
        s >> day_ >> zp;

        if (zp)
          zone_extract (s);
      }

      // gmonth
      //
      template <typename C, typename B>
      template <typename S>
      gmonth<C, B>::
      gmonth (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        bool zp;
        s >> month_ >> zp;

        if (zp)
          zone_extract (s);
      }

      // gyear
      //
      template <typename C, typename B>
      template <typename S>
      gyear<C, B>::
      gyear (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        bool zp;
        s >> year_ >> zp;

        if (zp)
          zone_extract (s);
      }

      // gmonth_day
      //
      template <typename C, typename B>
      template <typename S>
      gmonth_day<C, B>::
      gmonth_day (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        bool zp;
        s >> month_ >> day_ >> zp;

        if (zp)
          zone_extract (s);
      }

      // gyear_month
      //
      template <typename C, typename B>
      template <typename S>
      gyear_month<C, B>::
      gyear_month (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        bool zp;
        s >> year_ >> month_ >> zp;

        if (zp)
          zone_extract (s);
      }

      // date
      //
      template <typename C, typename B>
      template <typename S>
      date<C, B>::
      date (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        bool zp;
        s >> year_ >> month_ >> day_ >> zp;

        if (zp)
          zone_extract (s);
      }

      // time
      //
      template <typename C, typename B>
      template <typename S>
      time<C, B>::
      time (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        bool zp;
        s >> hours_ >> minutes_ >> seconds_ >> zp;

        if (zp)
          zone_extract (s);
      }

      // date_time
      //
      template <typename C, typename B>
      template <typename S>
      date_time<C, B>::
      date_time (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        bool zp;
        s >> year_ >> month_ >> day_
          >> hours_ >> minutes_ >> seconds_ >> zp;

        if (zp)
          zone_extract (s);
      }

      // duration
      //
      template <typename C, typename B>
      template <typename S>
      duration<C, B>::
      duration (istream<S>& s, flags f, container* c)
          : B (s, f, c)
      {
        s >> negative_
          >> years_ >> months_ >> days_
          >> hours_ >> minutes_ >> seconds_;
      }
    }
  }
}
