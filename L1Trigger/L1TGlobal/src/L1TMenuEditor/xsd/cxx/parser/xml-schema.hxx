// file      : xsd/cxx/parser/xml-schema.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_XML_SCHEMA_HXX
#define XSD_CXX_PARSER_XML_SCHEMA_HXX

#include <string>
#include <vector>
#include <cstddef> // std::size_t

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // String sequence. Used for the NMTOKENS and IDREFS types.
      //
      template <typename C>
      class string_sequence: public std::vector<std::basic_string<C> >
      {
      public:
        typedef std::basic_string<C> value_type;
        typedef std::vector<value_type> base;
        typedef typename base::size_type size_type;

        string_sequence ();

        explicit
        string_sequence (size_type n, const value_type& x = value_type ());

        template <typename I>
        string_sequence (const I& begin, const I& end);
      };

      template <typename C>
      bool
      operator== (const string_sequence<C>&, const string_sequence<C>&);

      template <typename C>
      bool
      operator!= (const string_sequence<C>&, const string_sequence<C>&);


      // QName
      //
      template <typename C>
      class qname
      {
      public:
        explicit
        qname (const std::basic_string<C>& name);

        qname (const std::basic_string<C>& prefix,
               const std::basic_string<C>& name);

        void
        swap (qname&);

        const std::basic_string<C>&
        prefix () const;

        std::basic_string<C>&
        prefix ();

        void
        prefix (const std::basic_string<C>&);

        const std::basic_string<C>&
        name () const;

        std::basic_string<C>&
        name ();

        void
        name (const std::basic_string<C>&);

      private:
        std::basic_string<C> prefix_;
        std::basic_string<C> name_;
      };

      template <typename C>
      bool
      operator== (const qname<C>&, const qname<C>&);

      template <typename C>
      bool
      operator!= (const qname<C>&, const qname<C>&);


      // Binary buffer. Used for the base64Binary and hexBinary types.
      //
      class buffer
      {
      public:
        typedef std::size_t size_t;

        class bounds {}; // Out of bounds exception.

      public:
        ~buffer ();

        explicit
        buffer (size_t size = 0);
        buffer (size_t size, size_t capacity);
        buffer (const void* data, size_t size);
        buffer (const void* data, size_t size, size_t capacity);

        // If the assume_ownership argument is true, the buffer will
        // assume the ownership of the data and will release the memory
        // by calling operator delete ().
        //
        buffer (void* data,
                size_t size,
                size_t capacity,
                bool assume_ownership);

        buffer (const buffer&);

      public:
        buffer&
        operator= (const buffer&);

      public:
        size_t
        capacity () const;

        // Returns true if the underlying buffer has moved.
        //
        bool
        capacity (size_t);

      public:
        size_t
        size () const;

        // Returns true if the underlying buffer has moved.
        //
        bool
        size (size_t);

      public:
        const char*
        data () const;

        char*
        data ();

        const char*
        begin () const;

        char*
        begin ();

        const char*
        end () const;

        char*
        end ();

      public:
        void
        swap (buffer&);

      private:
        bool
        capacity (size_t capacity, bool copy);

      private:
        char* data_;
        size_t size_;
        size_t capacity_;
      };

      bool
      operator== (const buffer&, const buffer&);

      bool
      operator!= (const buffer&, const buffer&);


      // Time and date types.
      //

      class time_zone
      {
      public:
        time_zone ();
        time_zone (short hours, short minutes);

        // Returns true if time zone is specified.
        //
        bool
        zone_present () const;

        // Resets the time zone to the 'not specified' state.
        //
        void
        zone_reset ();

        short
        zone_hours () const;

        void
        zone_hours (short);

        short
        zone_minutes () const;

        void
        zone_minutes (short);

      private:
        bool present_;
        short hours_;
        short minutes_;
      };

      bool
      operator== (const time_zone&, const time_zone&);

      bool
      operator!= (const time_zone&, const time_zone&);


      class gday: public time_zone
      {
      public:
        explicit
        gday (unsigned short day);
        gday (unsigned short day, short zone_hours, short zone_minutes);

        unsigned short
        day () const;

        void
        day (unsigned short);

      private:
        unsigned short day_;
      };

      bool
      operator== (const gday&, const gday&);

      bool
      operator!= (const gday&, const gday&);


      class gmonth: public time_zone
      {
      public:
        explicit
        gmonth (unsigned short month);
        gmonth (unsigned short month, short zone_hours, short zone_minutes);

        unsigned short
        month () const;

        void
        month (unsigned short);

      private:
        unsigned short month_;
      };

      bool
      operator== (const gmonth&, const gmonth&);

      bool
      operator!= (const gmonth&, const gmonth&);


      class gyear: public time_zone
      {
      public:
        explicit
        gyear (int year);
        gyear (int year, short zone_hours, short zone_minutes);

        int
        year () const;

        void
        year (int);

      private:
        int year_;
      };

      bool
      operator== (const gyear&, const gyear&);

      bool
      operator!= (const gyear&, const gyear&);


      class gmonth_day: public time_zone
      {
      public:
        gmonth_day (unsigned short month, unsigned short day);
        gmonth_day (unsigned short month, unsigned short day,
                    short zone_hours, short zone_minutes);

        unsigned short
        month () const;

        void
        month (unsigned short);

        unsigned short
        day () const;

        void
        day (unsigned short);

      private:
        unsigned short month_;
        unsigned short day_;
      };

      bool
      operator== (const gmonth_day&, const gmonth_day&);

      bool
      operator!= (const gmonth_day&, const gmonth_day&);


      class gyear_month: public time_zone
      {
      public:
        gyear_month (int year, unsigned short month);
        gyear_month (int year, unsigned short month,
                     short zone_hours, short zone_minutes);

        int
        year () const;

        void
        year (int);

        unsigned short
        month () const;

        void
        month (unsigned short);

      private:
        int year_;
        unsigned short month_;
      };

      bool
      operator== (const gyear_month&, const gyear_month&);

      bool
      operator!= (const gyear_month&, const gyear_month&);


      class date: public time_zone
      {
      public:
        date (int year, unsigned short month, unsigned short day);
        date (int year, unsigned short month, unsigned short day,
              short zone_hours, short zone_minutes);

        int
        year () const;

        void
        year (int);

        unsigned short
        month () const;

        void
        month (unsigned short);

        unsigned short
        day () const;

        void
        day (unsigned short);

      private:
        int year_;
        unsigned short month_;
        unsigned short day_;
      };

      bool
      operator== (const date&, const date&);

      bool
      operator!= (const date&, const date&);


      class time: public time_zone
      {
      public:
        time (unsigned short hours, unsigned short minutes, double seconds);
        time (unsigned short hours, unsigned short minutes, double seconds,
              short zone_hours, short zone_minutes);

        unsigned short
        hours () const;

        void
        hours (unsigned short);

        unsigned short
        minutes () const;

        void
        minutes (unsigned short);

        double
        seconds () const;

        void
        seconds (double);

      private:
        unsigned short hours_;
        unsigned short minutes_;
        double seconds_;
      };

      bool
      operator== (const time&, const time&);

      bool
      operator!= (const time&, const time&);


      class date_time: public time_zone
      {
      public:
        date_time (int year, unsigned short month, unsigned short day,
                   unsigned short hours, unsigned short minutes, double seconds);

        date_time (int year, unsigned short month, unsigned short day,
                   unsigned short hours, unsigned short minutes, double seconds,
                   short zone_hours, short zone_minutes);

        int
        year () const;

        void
        year (int);

        unsigned short
        month () const;

        void
        month (unsigned short);

        unsigned short
        day () const;

        void
        day (unsigned short);

        unsigned short
        hours () const;

        void
        hours (unsigned short);

        unsigned short
        minutes () const;

        void
        minutes (unsigned short);

        double
        seconds () const;

        void
        seconds (double);

      private:
        int year_;
        unsigned short month_;
        unsigned short day_;
        unsigned short hours_;
        unsigned short minutes_;
        double seconds_;
      };

      bool
      operator== (const date_time&, const date_time&);

      bool
      operator!= (const date_time&, const date_time&);


      class duration
      {
      public:
        duration (bool negative,
                  unsigned int years, unsigned int months, unsigned int days,
                  unsigned int hours, unsigned int minutes, double seconds);

        bool
        negative () const;

        void
        negative (bool);

        unsigned int
        years () const;

        void
        years (unsigned int);

        unsigned int
        months () const;

        void
        months (unsigned int);

        unsigned int
        days () const;

        void
        days (unsigned int);

        unsigned int
        hours () const;

        void
        hours (unsigned int);

        unsigned int
        minutes () const;

        void
        minutes (unsigned int);

        double
        seconds () const;

        void
        seconds (double);

      private:
        bool negative_;
        unsigned int years_;
        unsigned int months_;
        unsigned int days_;
        unsigned int hours_;
        unsigned int minutes_;
        double seconds_;
      };

      bool
      operator== (const duration&, const duration&);

      bool
      operator!= (const duration&, const duration&);
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/xml-schema.ixx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/xml-schema.txx>

#endif // XSD_CXX_PARSER_XML_SCHEMA_HXX
