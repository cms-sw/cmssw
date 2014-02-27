// file      : xsd/cxx/parser/xml-schema.ixx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <new>     // operator new/delete
#include <cstring> // std::memcpy, std::memcmp

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // string_sequence
      //
      template <typename C>
      string_sequence<C>::
      string_sequence ()
      {
      }

      template <typename C>
      string_sequence<C>::
      string_sequence (size_type n, const value_type& x)
          : base (n, x)
      {
      }

      template <typename C>
      template <typename I>
      string_sequence<C>::
      string_sequence (const I& begin, const I& end)
          : base (begin, end)
      {
      }

      template <typename C>
      inline bool
      operator!= (const string_sequence<C>& a, const string_sequence<C>& b)
      {
        return !(a == b);
      }

      // qname
      //
      template <typename C>
      inline qname<C>::
      qname (const std::basic_string<C>& name)
          : name_ (name)
      {
      }

      template <typename C>
      inline qname<C>::
      qname (const std::basic_string<C>& prefix,
             const std::basic_string<C>& name)
          : prefix_ (prefix), name_ (name)
      {
      }

      template <typename C>
      void qname<C>::
      swap (qname<C>& x)
      {
        prefix_.swap (x.prefix_);
        name_.swap (x.name_);
      }

      template <typename C>
      inline const std::basic_string<C>& qname<C>::
      prefix () const
      {
        return prefix_;
      }

      template <typename C>
      inline std::basic_string<C>& qname<C>::
      prefix ()
      {
        return prefix_;
      }

      template <typename C>
      inline void qname<C>::
      prefix (const std::basic_string<C>& prefix)
      {
        prefix_ = prefix;
      }

      template <typename C>
      inline const std::basic_string<C>& qname<C>::
      name () const
      {
        return name_;
      }

      template <typename C>
      inline std::basic_string<C>& qname<C>::
      name ()
      {
        return name_;
      }

      template <typename C>
      inline void qname<C>::
      name (const std::basic_string<C>& name)
      {
        name_ = name;
      }

      template <typename C>
      inline bool
      operator== (const qname<C>& a, const qname<C>& b)
      {
        return a.prefix () == b.prefix () && a.name () == b.name ();
      }

      template <typename C>
      inline bool
      operator!= (const qname<C>& a, const qname<C>& b)
      {
        return !(a == b);
      }

      // buffer
      //
      inline buffer::
      ~buffer ()
      {
        if (data_)
	  operator delete (data_);
      }

      inline buffer::
      buffer (size_t size)
          : data_ (0), size_ (0), capacity_ (0)
      {
        capacity (size);
        size_ = size;
      }

      inline buffer::
      buffer (size_t size, size_t cap)
          : data_ (0), size_ (0), capacity_ (0)
      {
        if (size > cap)
          throw bounds ();

        capacity (cap);
        size_ = size;
      }

      inline buffer::
      buffer (const void* data, size_t size)
          : data_ (0), size_ (0), capacity_ (0)
      {
        capacity (size);
        size_ = size;

        if (size_)
          std::memcpy (data_, data, size_);
      }

      inline buffer::
      buffer (const void* data, size_t size, size_t cap)
          : data_ (0), size_ (0), capacity_ (0)
      {
        if (size > cap)
          throw bounds ();

        capacity (cap);
        size_ = size;

        if (size_)
          std::memcpy (data_, data, size_);
      }

      inline buffer::
      buffer (void* data, size_t size, size_t cap, bool own)
          : data_ (0), size_ (0), capacity_ (0)
      {
        if (size > cap)
          throw bounds ();

        if (own)
        {
          data_ = reinterpret_cast<char*> (data);
          size_ = size;
          capacity_ = cap;
        }
        else
        {
          capacity (cap);
          size_ = size;

          if (size_)
            std::memcpy (data_, data, size_);
        }
      }

      inline buffer::
      buffer (const buffer& other)
          : data_ (0), size_ (0), capacity_ (0)
      {
        capacity (other.capacity_);
        size_ = other.size_;

        if (size_)
          std::memcpy (data_, other.data_, size_);
      }

      inline buffer& buffer::
      operator= (const buffer& other)
      {
        if (this != &other)
        {
          capacity (other.capacity_, false);
          size_ = other.size_;

          if (size_)
            std::memcpy (data_, other.data_, size_);
        }

        return *this;
      }

      inline size_t buffer::
      capacity () const
      {
        return capacity_;
      }

      inline bool buffer::
      capacity (size_t cap)
      {
        return capacity (cap, true);
      }

      inline size_t buffer::
      size () const
      {
        return size_;
      }

      inline bool buffer::
      size (size_t size)
      {
        bool r (false);

        if (size > capacity_)
          r = capacity (size);

        size_ = size;

        return r;
      }

      inline const char* buffer::
      data () const
      {
        return data_;
      }

      inline char* buffer::
      data ()
      {
        return data_;
      }

      inline const char* buffer::
      begin () const
      {
        return data_;
      }

      inline char* buffer::
      begin ()
      {
        return data_;
      }

      inline const char* buffer::
      end () const
      {
        return data_ + size_;
      }

      inline char* buffer::
      end ()
      {
        return data_ + size_;
      }

      inline void buffer::
      swap (buffer& other)
      {
        char* tmp_data (data_);
        size_t tmp_size (size_);
        size_t tmp_capacity (capacity_);

        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;

        other.data_ = tmp_data;
        other.size_ = tmp_size;
        other.capacity_ = tmp_capacity;
      }

      inline bool buffer::
      capacity (size_t capacity, bool copy)
      {
        if (size_ > capacity)
          throw bounds ();

        if (capacity <= capacity_)
        {
          return false; // Do nothing if shrinking is requested.
        }
        else
        {
          char* data (reinterpret_cast<char*> (operator new (capacity)));

          if (copy && size_ > 0)
            std::memcpy (data, data_, size_);

          if (data_)
            operator delete (data_);

          data_ = data;
          capacity_ = capacity;

          return true;
        }
      }

      inline bool
      operator== (const buffer& a, const buffer& b)
      {
        return a.size () == b.size () &&
          std::memcmp (a.data (), b.data (), a.size ()) == 0;
      }

      inline bool
      operator!= (const buffer& a, const buffer& b)
      {
        return !(a == b);
      }

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
      inline gday::
      gday (unsigned short day)
          : day_ (day)
      {
      }

      inline gday::
      gday (unsigned short day, short zh, short zm)
          : time_zone (zh, zm), day_ (day)
      {
      }

      inline unsigned short gday::
      day () const
      {
        return day_;
      }

      inline void gday::
      day (unsigned short day)
      {
        day_ = day;
      }

      inline bool
      operator== (const gday& a, const gday& b)
      {
        const time_zone& az = a;
        const time_zone& bz = b;

        return a.day () == b.day () && az == bz;
      }

      inline bool
      operator!= (const gday& a, const gday& b)
      {
        return !(a == b);
      }

      // gmonth
      //
      inline gmonth::
      gmonth (unsigned short month)
          : month_ (month)
      {
      }

      inline gmonth::
      gmonth (unsigned short month, short zh, short zm)
          : time_zone (zh, zm), month_ (month)
      {
      }

      inline unsigned short gmonth::
      month () const
      {
        return month_;
      }

      inline void gmonth::
      month (unsigned short month)
      {
        month_ = month;
      }

      inline bool
      operator== (const gmonth& a, const gmonth& b)
      {
        const time_zone& az = a;
        const time_zone& bz = b;

        return a.month () == b.month () && az == bz;
      }

      inline bool
      operator!= (const gmonth& a, const gmonth& b)
      {
        return !(a == b);
      }

      // gyear
      //
      inline gyear::
      gyear (int year)
          : year_ (year)
      {
      }

      inline gyear::
      gyear (int year, short zh, short zm)
          : time_zone (zh, zm), year_ (year)
      {
      }

      inline int gyear::
      year () const
      {
        return year_;
      }

      inline void gyear::
      year (int year)
      {
        year_ = year;
      }

      inline bool
      operator== (const gyear& a, const gyear& b)
      {
        const time_zone& az = a;
        const time_zone& bz = b;

        return a.year () == b.year () && az == bz;
      }

      inline bool
      operator!= (const gyear& a, const gyear& b)
      {
        return !(a == b);
      }

      // gmonth_day
      //
      inline gmonth_day::
      gmonth_day (unsigned short month, unsigned short day)
          : month_ (month), day_ (day)
      {
      }

      inline gmonth_day::
      gmonth_day (unsigned short month,
                  unsigned short day,
                  short zh, short zm)
          : time_zone (zh, zm), month_ (month), day_ (day)
      {
      }

      inline unsigned short gmonth_day::
      month () const
      {
        return month_;
      }

      inline void gmonth_day::
      month (unsigned short month)
      {
        month_ = month;
      }

      inline unsigned short gmonth_day::
      day () const
      {
        return day_;
      }

      inline void gmonth_day::
      day (unsigned short day)
      {
        day_ = day;
      }

      inline bool
      operator== (const gmonth_day& a, const gmonth_day& b)
      {
        const time_zone& az = a;
        const time_zone& bz = b;

        return a.month () == b.month () &&
          a.day () == b.day () &&
          az == bz;
      }

      inline bool
      operator!= (const gmonth_day& a, const gmonth_day& b)
      {
        return !(a == b);
      }

      // gyear_month
      //
      inline gyear_month::
      gyear_month (int year, unsigned short month)
          : year_ (year), month_ (month)
      {
      }

      inline gyear_month::
      gyear_month (int year, unsigned short month,
                   short zh, short zm)
          : time_zone (zh, zm), year_ (year), month_ (month)
      {
      }

      inline int gyear_month::
      year () const
      {
        return year_;
      }

      inline void gyear_month::
      year (int year)
      {
        year_ = year;
      }

      inline unsigned short gyear_month::
      month () const
      {
        return month_;
      }

      inline void gyear_month::
      month (unsigned short month)
      {
        month_ = month;
      }

      inline bool
      operator== (const gyear_month& a, const gyear_month& b)
      {
        const time_zone& az = a;
        const time_zone& bz = b;

        return a.year () == b.year () &&
          a.month () == b.month () &&
          az == bz;
      }

      inline bool
      operator!= (const gyear_month& a, const gyear_month& b)
      {
        return !(a == b);
      }

      // date
      //
      inline date::
      date (int year, unsigned short month, unsigned short day)
          : year_ (year), month_ (month), day_ (day)
      {
      }

      inline date::
      date (int year, unsigned short month, unsigned short day,
            short zh, short zm)
          : time_zone (zh, zm), year_ (year), month_ (month), day_ (day)
      {
      }

      inline int date::
      year () const
      {
        return year_;
      }

      inline void date::
      year (int year)
      {
        year_ = year;
      }

      inline unsigned short date::
      month () const
      {
        return month_;
      }

      inline void date::
      month (unsigned short month)
      {
        month_ = month;
      }

      inline unsigned short date::
      day () const
      {
        return day_;
      }

      inline void date::
      day (unsigned short day)
      {
        day_ = day;
      }

      inline bool
      operator== (const date& a, const date& b)
      {
        const time_zone& az = a;
        const time_zone& bz = b;

        return a.year () == b.year () &&
          a.month () == b.month () &&
          a.day () == b.day () &&
          az == bz;
      }

      inline bool
      operator!= (const date& a, const date& b)
      {
        return !(a == b);
      }

      // time
      //
      inline time::
      time (unsigned short hours, unsigned short minutes, double seconds)
          : hours_ (hours), minutes_ (minutes), seconds_ (seconds)
      {
      }

      inline time::
      time (unsigned short hours, unsigned short minutes, double seconds,
            short zh, short zm)
          : time_zone (zh, zm),
            hours_ (hours), minutes_ (minutes), seconds_ (seconds)
      {
      }

      inline unsigned short time::
      hours () const
      {
        return hours_;
      }

      inline void time::
      hours (unsigned short hours)
      {
        hours_ = hours;
      }

      inline unsigned short time::
      minutes () const
      {
        return minutes_;
      }

      inline void time::
      minutes (unsigned short minutes)
      {
        minutes_ = minutes;
      }

      inline double time::
      seconds () const
      {
        return seconds_;
      }

      inline void time::
      seconds (double seconds)
      {
        seconds_ = seconds;
      }

      inline bool
      operator== (const time& a, const time& b)
      {
        const time_zone& az = a;
        const time_zone& bz = b;

        return a.hours () == b.hours () &&
          a.minutes () == b.minutes () &&
          a.seconds () == b.seconds () &&
          az == bz;
      }

      inline bool
      operator!= (const time& a, const time& b)
      {
        return !(a == b);
      }

      // date_time
      //
      inline date_time::
      date_time (int year, unsigned short month, unsigned short day,
                 unsigned short hours, unsigned short minutes, double seconds)
          : year_ (year), month_ (month), day_ (day),
            hours_ (hours), minutes_ (minutes), seconds_ (seconds)
      {
      }

      inline date_time::
      date_time (int year, unsigned short month, unsigned short day,
                 unsigned short hours, unsigned short minutes, double seconds,
                 short zh, short zm)
          : time_zone (zh, zm),
            year_ (year), month_ (month), day_ (day),
            hours_ (hours), minutes_ (minutes), seconds_ (seconds)
      {
      }

      inline int date_time::
      year () const
      {
        return year_;
      }

      inline void date_time::
      year (int year)
      {
        year_ = year;
      }

      inline unsigned short date_time::
      month () const
      {
        return month_;
      }

      inline void date_time::
      month (unsigned short month)
      {
        month_ = month;
      }

      inline unsigned short date_time::
      day () const
      {
        return day_;
      }

      inline void date_time::
      day (unsigned short day)
      {
        day_ = day;
      }

      inline unsigned short date_time::
      hours () const
      {
        return hours_;
      }

      inline void date_time::
      hours (unsigned short hours)
      {
        hours_ = hours;
      }

      inline unsigned short date_time::
      minutes () const
      {
        return minutes_;
      }

      inline void date_time::
      minutes (unsigned short minutes)
      {
        minutes_ = minutes;
      }

      inline double date_time::
      seconds () const
      {
        return seconds_;
      }

      inline void date_time::
      seconds (double seconds)
      {
        seconds_ = seconds;
      }

      inline bool
      operator== (const date_time& a, const date_time& b)
      {
        const time_zone& az = a;
        const time_zone& bz = b;

        return a.year () == b.year () &&
          a.month () == b.month () &&
          a.day () == b.day () &&
          a.hours () == b.hours () &&
          a.minutes () == b.minutes () &&
          a.seconds () == b.seconds () &&
          az == bz;
      }

      inline bool
      operator!= (const date_time& a, const date_time& b)
      {
        return !(a == b);
      }

      // duration
      //
      inline duration::
      duration (bool negative,
                unsigned int years, unsigned int months, unsigned int days,
                unsigned int hours, unsigned int minutes, double seconds)
          : negative_ (negative),
            years_ (years), months_ (months), days_ (days),
            hours_ (hours), minutes_ (minutes), seconds_ (seconds)
      {
      }

      inline bool duration::
      negative () const
      {
        return negative_;
      }

      inline void duration::
      negative (bool negative)
      {
        negative_ = negative;
      }

      inline unsigned int duration::
      years () const
      {
        return years_;
      }

      inline void duration::
      years (unsigned int years)
      {
        years_ = years;
      }

      inline unsigned int duration::
      months () const
      {
        return months_;
      }

      inline void duration::
      months (unsigned int months)
      {
        months_ = months;
      }

      inline unsigned int duration::
      days () const
      {
        return days_;
      }

      inline void duration::
      days (unsigned int days)
      {
        days_ = days;
      }

      inline unsigned int duration::
      hours () const
      {
        return hours_;
      }

      inline void duration::
      hours (unsigned int hours)
      {
        hours_ = hours;
      }

      inline unsigned int duration::
      minutes () const
      {
        return minutes_;
      }

      inline void duration::
      minutes (unsigned int minutes)
      {
        minutes_ = minutes;
      }

      inline double duration::
      seconds () const
      {
        return seconds_;
      }

      inline void duration::
      seconds (double seconds)
      {
        seconds_ = seconds;
      }

      inline bool
      operator== (const duration& a, const duration& b)
      {
        return a.negative () == b.negative () &&
          a.years () == b.years () &&
          a.months () == b.months () &&
          a.days () == b.days () &&
          a.hours () == b.hours () &&
          a.minutes () == b.minutes () &&
          a.seconds () == b.seconds ();
      }

      inline bool
      operator!= (const duration& a, const duration& b)
      {
        return !(a == b);
      }
    }
  }
}
