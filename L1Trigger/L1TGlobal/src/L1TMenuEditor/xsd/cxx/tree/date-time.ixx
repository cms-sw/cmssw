// file      : xsd/cxx/tree/date-time.ixx
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
      inline time_zone::
      time_zone ()
          : present_ (false)
      {
      }

      inline time_zone::
      time_zone (short h, short m)
          : present_ (true), hours_ (h), minutes_ (m)
      {
      }

      inline bool time_zone::
      zone_present () const
      {
        return present_;
      }

      inline void time_zone::
      zone_reset ()
      {
        present_ = false;
      }

      inline short time_zone::
      zone_hours () const
      {
        return hours_;
      }

      inline void time_zone::
      zone_hours (short h)
      {
        hours_ = h;
        present_ = true;
      }

      inline short time_zone::
      zone_minutes () const
      {
        return minutes_;
      }

      inline void time_zone::
      zone_minutes (short m)
      {
        minutes_ = m;
        present_ = true;
      }

      inline bool
      operator== (const time_zone& x, const time_zone& y)
      {
        return x.zone_present ()
          ? y.zone_present () &&
          x.zone_hours () == y.zone_hours () &&
          x.zone_minutes () == y.zone_minutes ()
          : !y.zone_present ();
      }

      inline bool
      operator!= (const time_zone& x, const time_zone& y)
      {
        return !(x == y);
      }

      // gday
      //
      template <typename C, typename B>
      inline gday<C, B>::
      gday ()
      {
      }

      template <typename C, typename B>
      inline gday<C, B>::
      gday (unsigned short day)
          : day_ (day)
      {
      }

      template <typename C, typename B>
      inline gday<C, B>::
      gday (unsigned short day, short zone_h, short zone_m)
          : time_zone (zone_h, zone_m), day_ (day)
      {
      }

      template <typename C, typename B>
      inline gday<C, B>::
      gday (const gday& x, flags f, container* c)
          : B (x, f, c), time_zone (x), day_ (x.day_)
      {
      }

      template <typename C, typename B>
      inline unsigned short gday<C, B>::
      day () const
      {
        return day_;
      }

      template <typename C, typename B>
      inline void gday<C, B>::
      day (unsigned short day)
      {
        day_ = day;
      }

      template <typename C, typename B>
      inline bool
      operator== (const gday<C, B>& x, const gday<C, B>& y)
      {
        const time_zone& xz = x;
        const time_zone& yz = y;

        return x.day () == y.day () && xz == yz;
      }

      template <typename C, typename B>
      inline bool
      operator!= (const gday<C, B>& x, const gday<C, B>& y)
      {
        return !(x == y);
      }

      // gmonth
      //
      template <typename C, typename B>
      inline gmonth<C, B>::
      gmonth ()
      {
      }

      template <typename C, typename B>
      inline gmonth<C, B>::
      gmonth (unsigned short month)
          : month_ (month)
      {
      }

      template <typename C, typename B>
      inline gmonth<C, B>::
      gmonth (unsigned short month, short zone_h, short zone_m)
          : time_zone (zone_h, zone_m), month_ (month)
      {
      }

      template <typename C, typename B>
      inline gmonth<C, B>::
      gmonth (const gmonth& x, flags f, container* c)
          : B (x, f, c), time_zone (x), month_ (x.month_)
      {
      }

      template <typename C, typename B>
      inline unsigned short gmonth<C, B>::
      month () const
      {
        return month_;
      }

      template <typename C, typename B>
      inline void gmonth<C, B>::
      month (unsigned short month)
      {
        month_ = month;
      }

      template <typename C, typename B>
      inline bool
      operator== (const gmonth<C, B>& x, const gmonth<C, B>& y)
      {
        const time_zone& xz = x;
        const time_zone& yz = y;

        return x.month () == y.month () && xz == yz;
      }

      template <typename C, typename B>
      inline bool
      operator!= (const gmonth<C, B>& x, const gmonth<C, B>& y)
      {
        return !(x == y);
      }

      // gyear
      //
      template <typename C, typename B>
      inline gyear<C, B>::
      gyear ()
      {
      }

      template <typename C, typename B>
      inline gyear<C, B>::
      gyear (int year)
          : year_ (year)
      {
      }

      template <typename C, typename B>
      inline gyear<C, B>::
      gyear (int year, short zone_h, short zone_m)
          : time_zone (zone_h, zone_m), year_ (year)
      {
      }

      template <typename C, typename B>
      inline gyear<C, B>::
      gyear (const gyear& x, flags f, container* c)
          : B (x, f, c), time_zone (x), year_ (x.year_)
      {
      }

      template <typename C, typename B>
      inline int gyear<C, B>::
      year () const
      {
        return year_;
      }

      template <typename C, typename B>
      inline void gyear<C, B>::
      year (int year)
      {
        year_ = year;
      }

      template <typename C, typename B>
      inline bool
      operator== (const gyear<C, B>& x, const gyear<C, B>& y)
      {
        const time_zone& xz = x;
        const time_zone& yz = y;

        return x.year () == y.year () && xz == yz;
      }

      template <typename C, typename B>
      inline bool
      operator!= (const gyear<C, B>& x, const gyear<C, B>& y)
      {
        return !(x == y);
      }

      // gmonth_day
      //
      template <typename C, typename B>
      inline gmonth_day<C, B>::
      gmonth_day ()
      {
      }

      template <typename C, typename B>
      inline gmonth_day<C, B>::
      gmonth_day (unsigned short month, unsigned short day)
          : month_ (month), day_ (day)
      {
      }

      template <typename C, typename B>
      inline gmonth_day<C, B>::
      gmonth_day (unsigned short month, unsigned short day,
                  short zone_h, short zone_m)
          : time_zone (zone_h, zone_m), month_ (month), day_ (day)
      {
      }

      template <typename C, typename B>
      inline gmonth_day<C, B>::
      gmonth_day (const gmonth_day& x, flags f, container* c)
          : B (x, f, c), time_zone (x), month_ (x.month_), day_ (x.day_)
      {
      }

      template <typename C, typename B>
      inline unsigned short gmonth_day<C, B>::
      month () const
      {
        return month_;
      }

      template <typename C, typename B>
      inline void gmonth_day<C, B>::
      month (unsigned short month)
      {
        month_ = month;
      }

      template <typename C, typename B>
      inline unsigned short gmonth_day<C, B>::
      day () const
      {
        return day_;
      }

      template <typename C, typename B>
      inline void gmonth_day<C, B>::
      day (unsigned short day)
      {
        day_ = day;
      }

      template <typename C, typename B>
      inline bool
      operator== (const gmonth_day<C, B>& x, const gmonth_day<C, B>& y)
      {
        const time_zone& xz = x;
        const time_zone& yz = y;

        return x.month () == y.month () &&
          x.day () == y.day () &&
          xz == yz;
      }

      template <typename C, typename B>
      inline bool
      operator!= (const gmonth_day<C, B>& x, const gmonth_day<C, B>& y)
      {
        return !(x == y);
      }

      // gyear_month
      //
      template <typename C, typename B>
      inline gyear_month<C, B>::
      gyear_month ()
      {
      }

      template <typename C, typename B>
      inline gyear_month<C, B>::
      gyear_month (int year, unsigned short month)
          : year_ (year), month_ (month)
      {
      }

      template <typename C, typename B>
      inline gyear_month<C, B>::
      gyear_month (int year, unsigned short month,
                   short zone_h, short zone_m)
          : time_zone (zone_h, zone_m), year_ (year), month_ (month)
      {
      }

      template <typename C, typename B>
      inline gyear_month<C, B>::
      gyear_month (const gyear_month& x, flags f, container* c)
          : B (x, f, c), time_zone (x), year_ (x.year_), month_ (x.month_)
      {
      }

      template <typename C, typename B>
      inline int gyear_month<C, B>::
      year () const
      {
        return year_;
      }

      template <typename C, typename B>
      inline void gyear_month<C, B>::
      year (int year)
      {
        year_ = year;
      }

      template <typename C, typename B>
      inline unsigned short gyear_month<C, B>::
      month () const
      {
        return month_;
      }

      template <typename C, typename B>
      inline void gyear_month<C, B>::
      month (unsigned short month)
      {
        month_ = month;
      }

      template <typename C, typename B>
      inline bool
      operator== (const gyear_month<C, B>& x, const gyear_month<C, B>& y)
      {
        const time_zone& xz = x;
        const time_zone& yz = y;

        return x.year () == y.year () &&
          x.month () == y.month () &&
          xz == yz;
      }

      template <typename C, typename B>
      inline bool
      operator!= (const gyear_month<C, B>& x, const gyear_month<C, B>& y)
      {
        return !(x == y);
      }

      // date
      //
      template <typename C, typename B>
      inline date<C, B>::
      date ()
      {
      }

      template <typename C, typename B>
      inline date<C, B>::
      date (int year, unsigned short month, unsigned short day)
          : year_ (year), month_ (month), day_ (day)
      {
      }

      template <typename C, typename B>
      inline date<C, B>::
      date (int year, unsigned short month, unsigned short day,
            short zone_h, short zone_m)
          : time_zone (zone_h, zone_m),
            year_ (year), month_ (month), day_ (day)
      {
      }

      template <typename C, typename B>
      inline date<C, B>::
      date (const date& x, flags f, container* c)
          : B (x, f, c), time_zone (x),
            year_ (x.year_), month_ (x.month_), day_ (x.day_)
      {
      }

      template <typename C, typename B>
      inline int date<C, B>::
      year () const
      {
        return year_;
      }

      template <typename C, typename B>
      inline void date<C, B>::
      year (int year)
      {
        year_ = year;
      }

      template <typename C, typename B>
      inline unsigned short date<C, B>::
      month () const
      {
        return month_;
      }

      template <typename C, typename B>
      inline void date<C, B>::
      month (unsigned short month)
      {
        month_ = month;
      }

      template <typename C, typename B>
      inline unsigned short date<C, B>::
      day () const
      {
        return day_;
      }

      template <typename C, typename B>
      inline void date<C, B>::
      day (unsigned short day)
      {
        day_ = day;
      }

      template <typename C, typename B>
      inline bool
      operator== (const date<C, B>& x, const date<C, B>& y)
      {
        const time_zone& xz = x;
        const time_zone& yz = y;

        return x.year () == y.year () &&
          x.month () == y.month () &&
          x.day () == y.day () &&
          xz == yz;
      }

      template <typename C, typename B>
      inline bool
      operator!= (const date<C, B>& x, const date<C, B>& y)
      {
        return !(x == y);
      }

      // time
      //
      template <typename C, typename B>
      inline time<C, B>::
      time ()
      {
      }

      template <typename C, typename B>
      inline time<C, B>::
      time (unsigned short hours, unsigned short minutes, double seconds)
          : hours_ (hours), minutes_ (minutes), seconds_ (seconds)
      {
      }

      template <typename C, typename B>
      inline time<C, B>::
      time (unsigned short hours, unsigned short minutes, double seconds,
            short zone_h, short zone_m)
          : time_zone (zone_h, zone_m),
            hours_ (hours), minutes_ (minutes), seconds_ (seconds)
      {
      }

      template <typename C, typename B>
      inline time<C, B>::
      time (const time& x, flags f, container* c)
          : B (x, f, c), time_zone (x),
            hours_ (x.hours_), minutes_ (x.minutes_), seconds_ (x.seconds_)
      {
      }

      template <typename C, typename B>
      inline unsigned short time<C, B>::
      hours () const
      {
        return hours_;
      }

      template <typename C, typename B>
      inline void time<C, B>::
      hours (unsigned short hours)
      {
        hours_ = hours;
      }

      template <typename C, typename B>
      inline unsigned short time<C, B>::
      minutes () const
      {
        return minutes_;
      }

      template <typename C, typename B>
      inline void time<C, B>::
      minutes (unsigned short minutes)
      {
        minutes_ = minutes;
      }

      template <typename C, typename B>
      inline double time<C, B>::
      seconds () const
      {
        return seconds_;
      }

      template <typename C, typename B>
      inline void time<C, B>::
      seconds (double seconds)
      {
        seconds_ = seconds;
      }

      template <typename C, typename B>
      inline bool
      operator== (const time<C, B>& x, const time<C, B>& y)
      {
        const time_zone& xz = x;
        const time_zone& yz = y;

        return x.hours () == y.hours () &&
          x.minutes () == y.minutes () &&
          x.seconds () == y.seconds () &&
          xz == yz;
      }

      template <typename C, typename B>
      inline bool
      operator!= (const time<C, B>& x, const time<C, B>& y)
      {
        return !(x == y);
      }

      // date_time
      //
      template <typename C, typename B>
      inline date_time<C, B>::
      date_time ()
      {
      }

      template <typename C, typename B>
      inline date_time<C, B>::
      date_time (int year, unsigned short month, unsigned short day,
                 unsigned short hours, unsigned short minutes, double seconds)
          : year_ (year), month_ (month), day_ (day),
            hours_ (hours), minutes_ (minutes), seconds_ (seconds)
      {
      }

      template <typename C, typename B>
      inline date_time<C, B>::
      date_time (int year, unsigned short month, unsigned short day,
                 unsigned short hours, unsigned short minutes, double seconds,
                 short zone_h, short zone_m)
          : time_zone (zone_h, zone_m),
            year_ (year), month_ (month), day_ (day),
            hours_ (hours), minutes_ (minutes), seconds_ (seconds)
      {
      }

      template <typename C, typename B>
      inline date_time<C, B>::
      date_time (const date_time& x, flags f, container* c)
          : B (x, f, c), time_zone (x),
            year_ (x.year_), month_ (x.month_), day_ (x.day_),
            hours_ (x.hours_), minutes_ (x.minutes_), seconds_ (x.seconds_)
      {
      }

      template <typename C, typename B>
      inline int date_time<C, B>::
      year () const
      {
        return year_;
      }

      template <typename C, typename B>
      inline void date_time<C, B>::
      year (int year)
      {
        year_ = year;
      }

      template <typename C, typename B>
      inline unsigned short date_time<C, B>::
      month () const
      {
        return month_;
      }

      template <typename C, typename B>
      inline void date_time<C, B>::
      month (unsigned short month)
      {
        month_ = month;
      }

      template <typename C, typename B>
      inline unsigned short date_time<C, B>::
      day () const
      {
        return day_;
      }

      template <typename C, typename B>
      inline void date_time<C, B>::
      day (unsigned short day)
      {
        day_ = day;
      }

      template <typename C, typename B>
      inline unsigned short date_time<C, B>::
      hours () const
      {
        return hours_;
      }

      template <typename C, typename B>
      inline void date_time<C, B>::
      hours (unsigned short hours)
      {
        hours_ = hours;
      }

      template <typename C, typename B>
      inline unsigned short date_time<C, B>::
      minutes () const
      {
        return minutes_;
      }

      template <typename C, typename B>
      inline void date_time<C, B>::
      minutes (unsigned short minutes)
      {
        minutes_ = minutes;
      }

      template <typename C, typename B>
      inline double date_time<C, B>::
      seconds () const
      {
        return seconds_;
      }

      template <typename C, typename B>
      inline void date_time<C, B>::
      seconds (double seconds)
      {
        seconds_ = seconds;
      }

      template <typename C, typename B>
      inline bool
      operator== (const date_time<C, B>& x, const date_time<C, B>& y)
      {
        const time_zone& xz = x;
        const time_zone& yz = y;

        return x.year () == y.year () &&
          x.month () == y.month () &&
          x.day () == y.day () &&
          x.hours () == y.hours () &&
          x.minutes () == y.minutes () &&
          x.seconds () == y.seconds () &&
          xz == yz;
      }

      template <typename C, typename B>
      inline bool
      operator!= (const date_time<C, B>& x, const date_time<C, B>& y)
      {
        return !(x == y);
      }

      // duration
      //
      template <typename C, typename B>
      inline duration<C, B>::
      duration ()
      {
      }

      template <typename C, typename B>
      inline duration<C, B>::
      duration (bool negative,
                unsigned int years, unsigned int months, unsigned int days,
                unsigned int hours, unsigned int minutes, double seconds)
          : negative_ (negative),
            years_ (years), months_ (months), days_ (days),
            hours_ (hours), minutes_ (minutes), seconds_ (seconds)
      {
      }

      template <typename C, typename B>
      inline duration<C, B>::
      duration (const duration& x, flags f, container* c)
          : B (x, f, c), negative_ (x.negative_),
            years_ (x.years_), months_ (x.months_), days_ (x.days_),
            hours_ (x.hours_), minutes_ (x.minutes_), seconds_ (x.seconds_)
      {
      }

      template <typename C, typename B>
      inline bool duration<C, B>::
      negative () const
      {
        return negative_;
      }

      template <typename C, typename B>
      inline void duration<C, B>::
      negative (bool negative)
      {
        negative_ = negative;
      }

      template <typename C, typename B>
      inline unsigned int duration<C, B>::
      years () const
      {
        return years_;
      }

      template <typename C, typename B>
      inline void duration<C, B>::
      years (unsigned int years)
      {
        years_ = years;
      }

      template <typename C, typename B>
      inline unsigned int duration<C, B>::
      months () const
      {
        return months_;
      }

      template <typename C, typename B>
      inline void duration<C, B>::
      months (unsigned int months)
      {
        months_ = months;
      }

      template <typename C, typename B>
      inline unsigned int duration<C, B>::
      days () const
      {
        return days_;
      }

      template <typename C, typename B>
      inline void duration<C, B>::
      days (unsigned int days)
      {
        days_ = days;
      }

      template <typename C, typename B>
      inline unsigned int duration<C, B>::
      hours () const
      {
        return hours_;
      }

      template <typename C, typename B>
      inline void duration<C, B>::
      hours (unsigned int hours)
      {
        hours_ = hours;
      }

      template <typename C, typename B>
      inline unsigned int duration<C, B>::
      minutes () const
      {
        return minutes_;
      }

      template <typename C, typename B>
      inline void duration<C, B>::
      minutes (unsigned int minutes)
      {
        minutes_ = minutes;
      }

      template <typename C, typename B>
      inline double duration<C, B>::
      seconds () const
      {
        return seconds_;
      }

      template <typename C, typename B>
      inline void duration<C, B>::
      seconds (double seconds)
      {
        seconds_ = seconds;
      }

      template <typename C, typename B>
      inline bool
      operator== (const duration<C, B>& x, const duration<C, B>& y)
      {
        return x.negative () == y.negative () &&
          x.years () == y.years () &&
          x.months () == y.months () &&
          x.days () == y.days () &&
          x.hours () == y.hours () &&
          x.minutes () == y.minutes () &&
          x.seconds () == y.seconds ();
      }

      template <typename C, typename B>
      inline bool
      operator!= (const duration<C, B>& x, const duration<C, B>& y)
      {
        return !(x == y);
      }
    }
  }
}
