// file      : xsd/cxx/tree/date-time.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

/**
 * @file
 *
 * @brief Contains C++ class definitions for the XML Schema date/time types.
 *
 * This is an internal header and is included by the generated code. You
 * normally should not include it directly.
 *
 */

#ifndef XSD_CXX_TREE_DATE_TIME_HXX
#define XSD_CXX_TREE_DATE_TIME_HXX

#include <string>
#include <cstddef> // std::size_t

#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMElement.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/istream-fwd.hxx>

namespace xsd
{
  namespace cxx
  {
    /**
     * @brief C++/Tree mapping runtime namespace.
     *
     * This is an internal namespace and normally should not be referenced
     * directly. Instead you should use the aliases for types in this
     * namespaces that are created in the generated code.
     *
     */
    namespace tree
    {
      /**
       * @brief Time zone representation
       *
       * The %time_zone class represents an optional %time zone and
       * is used as a base class for date/time types.
       *
       * The %time zone can negative in which case both the hours and
       * minutes components should be negative.
       *
       * @nosubgrouping
       */
      class time_zone
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Default constructor.
         *
         * This constructor initializes the instance to the 'not specified'
         * state.
         */
        time_zone ();

        /**
         * @brief Initialize an instance with the hours and minutes
         * components.
         *
         * @param hours The %time zone hours component.
         * @param minutes The %time zone minutes component.
         */
        time_zone (short hours, short minutes);

        //@}

        /**
         * @brief Determine if %time zone is specified.
         *
         * @return True if %time zone is specified, false otherwise.
         */
        bool
        zone_present () const;

        /**
         * @brief Reset the %time zone to the 'not specified' state.
         *
         */
        void
        zone_reset ();

        /**
         * @brief Get the hours component of the %time zone.
         *
         * @return The hours component of the %time zone.
         */
        short
        zone_hours () const;

        /**
         * @brief Set the hours component of the %time zone.
         *
         * @param h The new hours component.
         */
        void
        zone_hours (short h);


        /**
         * @brief Get the minutes component of the %time zone.
         *
         * @return The minutes component of the %time zone.
         */
        short
        zone_minutes () const;

        /**
         * @brief Set the minutes component of the %time zone.
         *
         * @param m The new minutes component.
         */
        void
        zone_minutes (short m);

      protected:
        //@cond

        template <typename C>
        void
        zone_parse (const C*, std::size_t);

        template <typename S>
        void
        zone_extract (istream<S>&);

        //@endcond

      private:
        bool present_;
        short hours_;
        short minutes_;
      };

      /**
       * @brief %time_zone comparison operator.
       *
       * @return True if both %time zones are either not specified or
       * have equal hours and minutes components, false otherwise.
       */
      bool
      operator== (const time_zone&, const time_zone&);

      /**
       * @brief %time_zone comparison operator.
       *
       * @return False if both %time zones are either not specified or
       * have equal hours and minutes components, true otherwise.
       */
      bool
      operator!= (const time_zone&, const time_zone&);


      /**
       * @brief Class corresponding to the XML Schema gDay built-in type.
       *
       * The %gday class represents a day of the month with an optional
       * %time zone.
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class gday: public B, public time_zone
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with the day component.
         *
         * When this constructor is used, the %time zone is left
         * unspecified.
         *
         * @param day The day component.
         */
        explicit
        gday (unsigned short day);

        /**
         * @brief Initialize an instance with the day component and %time
         * zone.
         *
         * @param day The day component.
         * @param zone_hours The %time zone hours component.
         * @param zone_minutes The %time zone minutes component.
         */
        gday (unsigned short day, short zone_hours, short zone_minutes);

        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the _clone function instead.
         */
        gday (const gday& x, flags f = 0, container* c = 0);

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual gday*
        _clone (flags f = 0, container* c = 0) const;

        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        gday (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gday (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gday (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gday (const std::basic_string<C>& s,
              const xercesc::DOMElement* e,
              flags f = 0,
              container* c = 0);
        //@}

      public:
        /**
         * @brief Get the day component.
         *
         * @return The day component.
         */
        unsigned short
        day () const;

        /**
         * @brief Set the day component.
         *
         * @param d The new day component.
         */
        void
        day (unsigned short d);

      protected:
        //@cond

        gday ();

        void
        parse (const std::basic_string<C>&);

        //@endcond

      private:
        unsigned short day_;
      };

      /**
       * @brief %gday comparison operator.
       *
       * @return True if the day components and %time zones are equal, false
       * otherwise.
       */
      template <typename C, typename B>
      bool
      operator== (const gday<C, B>&, const gday<C, B>&);

      /**
       * @brief %gday comparison operator.
       *
       * @return False if the day components and %time zones are equal, true
       * otherwise.
       */
      template <typename C, typename B>
      bool
      operator!= (const gday<C, B>&, const gday<C, B>&);

      /**
       * @brief Class corresponding to the XML Schema gMonth built-in type.
       *
       * The %gmonth class represents a month of the year with an optional
       * %time zone.
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class gmonth: public B, public time_zone
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with the month component.
         *
         * When this constructor is used, the %time zone is left
         * unspecified.
         *
         * @param month The month component.
         */
        explicit
        gmonth (unsigned short month);

        /**
         * @brief Initialize an instance with the month component and %time
         * zone.
         *
         * @param month The month component.
         * @param zone_hours The %time zone hours component.
         * @param zone_minutes The %time zone minutes component.
         */
        gmonth (unsigned short month, short zone_hours, short zone_minutes);

        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the _clone function instead.
         */
        gmonth (const gmonth& x, flags f = 0, container* c = 0);

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual gmonth*
        _clone (flags f = 0, container* c = 0) const;

        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        gmonth (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gmonth (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gmonth (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gmonth (const std::basic_string<C>& s,
                const xercesc::DOMElement* e,
                flags f = 0,
                container* c = 0);
        //@}

      public:
        /**
         * @brief Get the month component.
         *
         * @return The month component.
         */
        unsigned short
        month () const;

        /**
         * @brief Set the month component.
         *
         * @param m The new month component.
         */
        void
        month (unsigned short m);

      protected:
        //@cond

        gmonth ();

        void
        parse (const std::basic_string<C>&);

        //@endcond

      private:
        unsigned short month_;
      };

      /**
       * @brief %gmonth comparison operator.
       *
       * @return True if the month components and %time zones are equal, false
       * otherwise.
       */
      template <typename C, typename B>
      bool
      operator== (const gmonth<C, B>&, const gmonth<C, B>&);

      /**
       * @brief %gmonth comparison operator.
       *
       * @return False if the month components and %time zones are equal, true
       * otherwise.
       */
      template <typename C, typename B>
      bool
      operator!= (const gmonth<C, B>&, const gmonth<C, B>&);


      /**
       * @brief Class corresponding to the XML Schema gYear built-in type.
       *
       * The %gyear class represents a year with an optional %time zone.
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class gyear: public B, public time_zone
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with the year component.
         *
         * When this constructor is used, the %time zone is left
         * unspecified.
         *
         * @param year The year component.
         */
        explicit
        gyear (int year);

        /**
         * @brief Initialize an instance with the year component and %time
         * zone.
         *
         * @param year The year component.
         * @param zone_hours The %time zone hours component.
         * @param zone_minutes The %time zone minutes component.
         */
        gyear (int year, short zone_hours, short zone_minutes);

        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the _clone function instead.
         */
        gyear (const gyear& x, flags f = 0, container* c = 0);

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual gyear*
        _clone (flags f = 0, container* c = 0) const;

        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        gyear (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gyear (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gyear (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gyear (const std::basic_string<C>& s,
               const xercesc::DOMElement* e,
               flags f = 0,
               container* c = 0);
        //@}

      public:
        /**
         * @brief Get the year component.
         *
         * @return The year component.
         */
        int
        year () const;

        /**
         * @brief Set the year component.
         *
         * @param y The new year component.
         */
        void
        year (int y);

      protected:
        //@cond

        gyear ();

        void
        parse (const std::basic_string<C>&);

        //@endcond

      private:
        int year_;
      };

      /**
       * @brief %gyear comparison operator.
       *
       * @return True if the year components and %time zones are equal, false
       * otherwise.
       */
      template <typename C, typename B>
      bool
      operator== (const gyear<C, B>&, const gyear<C, B>&);

      /**
       * @brief %gyear comparison operator.
       *
       * @return False if the year components and %time zones are equal, true
       * otherwise.
       */
      template <typename C, typename B>
      bool
      operator!= (const gyear<C, B>&, const gyear<C, B>&);


      /**
       * @brief Class corresponding to the XML Schema gMonthDay built-in type.
       *
       * The %gmonth_day class represents day and month of the year with an
       * optional %time zone.
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class gmonth_day: public B, public time_zone
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with the month and day components.
         *
         * When this constructor is used, the %time zone is left
         * unspecified.
         *
         * @param month The month component.
         * @param day The day component.
         */
        gmonth_day (unsigned short month, unsigned short day);

        /**
         * @brief Initialize an instance with the month and day components
         * as well as %time zone.
         *
         * @param month The month component.
         * @param day The day component.
         * @param zone_hours The %time zone hours component.
         * @param zone_minutes The %time zone minutes component.
         */
        gmonth_day (unsigned short month, unsigned short day,
                    short zone_hours, short zone_minutes);

        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the _clone function instead.
         */
        gmonth_day (const gmonth_day& x, flags f = 0, container* c = 0);

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual gmonth_day*
        _clone (flags f = 0, container* c = 0) const;

        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        gmonth_day (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gmonth_day (const xercesc::DOMElement& e,
                    flags f = 0,
                    container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gmonth_day (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gmonth_day (const std::basic_string<C>& s,
                    const xercesc::DOMElement* e,
                    flags f = 0,
                    container* c = 0);
        //@}

      public:
        /**
         * @brief Get the month component.
         *
         * @return The month component.
         */
        unsigned short
        month () const;

        /**
         * @brief Set the month component.
         *
         * @param m The new month component.
         */
        void
        month (unsigned short m);

        /**
         * @brief Get the day component.
         *
         * @return The day component.
         */
        unsigned short
        day () const;

        /**
         * @brief Set the day component.
         *
         * @param d The new day component.
         */
        void
        day (unsigned short d);

      protected:
        //@cond

        gmonth_day ();

        void
        parse (const std::basic_string<C>&);

        //@endcond

      private:
        unsigned short month_;
        unsigned short day_;
      };

      /**
       * @brief %gmonth_day comparison operator.
       *
       * @return True if the month and day components as well as %time zones
       * are equal, false otherwise.
       */
      template <typename C, typename B>
      bool
      operator== (const gmonth_day<C, B>&, const gmonth_day<C, B>&);

      /**
       * @brief %gmonth_day comparison operator.
       *
       * @return False if the month and day components as well as %time zones
       * are equal, true otherwise.
       */
      template <typename C, typename B>
      bool
      operator!= (const gmonth_day<C, B>&, const gmonth_day<C, B>&);


      /**
       * @brief Class corresponding to the XML Schema gYearMonth built-in
       * type.
       *
       * The %gyear_month class represents year and month with an optional
       * %time zone.
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class gyear_month: public B, public time_zone
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with the year and month components.
         *
         * When this constructor is used, the %time zone is left
         * unspecified.
         *
         * @param year The year component.
         * @param month The month component.
         */
        gyear_month (int year, unsigned short month);

        /**
         * @brief Initialize an instance with the year and month components
         * as well as %time zone.
         *
         * @param year The year component.
         * @param month The month component.
         * @param zone_hours The %time zone hours component.
         * @param zone_minutes The %time zone minutes component.
         */
        gyear_month (int year, unsigned short month,
                     short zone_hours, short zone_minutes);

        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the _clone function instead.
         */
        gyear_month (const gyear_month& x, flags f = 0, container* c = 0);

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual gyear_month*
        _clone (flags f = 0, container* c = 0) const;

        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        gyear_month (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gyear_month (const xercesc::DOMElement& e,
                     flags f = 0,
                     container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gyear_month (const xercesc::DOMAttr& a,
                     flags f = 0,
                     container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        gyear_month (const std::basic_string<C>& s,
                     const xercesc::DOMElement* e,
                     flags f = 0,
                     container* c = 0);
        //@}

      public:
        /**
         * @brief Get the year component.
         *
         * @return The year component.
         */
        int
        year () const;

        /**
         * @brief Set the year component.
         *
         * @param y The new year component.
         */
        void
        year (int y);

        /**
         * @brief Get the month component.
         *
         * @return The month component.
         */
        unsigned short
        month () const;

        /**
         * @brief Set the month component.
         *
         * @param m The new month component.
         */
        void
        month (unsigned short m);

      protected:
        //@cond

        gyear_month ();

        void
        parse (const std::basic_string<C>&);

        //@endcond

      private:
        int year_;
        unsigned short month_;
      };

      /**
       * @brief %gyear_month comparison operator.
       *
       * @return True if the year and month components as well as %time zones
       * are equal, false otherwise.
       */
      template <typename C, typename B>
      bool
      operator== (const gyear_month<C, B>&, const gyear_month<C, B>&);

      /**
       * @brief %gyear_month comparison operator.
       *
       * @return False if the year and month components as well as %time zones
       * are equal, true otherwise.
       */
      template <typename C, typename B>
      bool
      operator!= (const gyear_month<C, B>&, const gyear_month<C, B>&);


      /**
       * @brief Class corresponding to the XML Schema %date built-in type.
       *
       * The %date class represents day, month, and year with an optional
       * %time zone.
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class date: public B, public time_zone
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with the year, month, and day
         * components.
         *
         * When this constructor is used, the %time zone is left
         * unspecified.
         *
         * @param year The year component.
         * @param month The month component.
         * @param day The day component.
         */
        date (int year, unsigned short month, unsigned short day);

        /**
         * @brief Initialize an instance with the year, month, and day
         * components as well as %time zone.
         *
         * @param year The year component.
         * @param month The month component.
         * @param day The day component.
         * @param zone_hours The %time zone hours component.
         * @param zone_minutes The %time zone minutes component.
         */
        date (int year, unsigned short month, unsigned short day,
              short zone_hours, short zone_minutes);

        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the _clone function instead.
         */
        date (const date& x, flags f = 0, container* c = 0);

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual date*
        _clone (flags f = 0, container* c = 0) const;

        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        date (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        date (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        date (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        date (const std::basic_string<C>& s,
              const xercesc::DOMElement* e,
              flags f = 0,
              container* c = 0);
        //@}

      public:
        /**
         * @brief Get the year component.
         *
         * @return The year component.
         */
        int
        year () const;

        /**
         * @brief Set the year component.
         *
         * @param y The new year component.
         */
        void
        year (int y);

        /**
         * @brief Get the month component.
         *
         * @return The month component.
         */
        unsigned short
        month () const;

        /**
         * @brief Set the month component.
         *
         * @param m The new month component.
         */
        void
        month (unsigned short m);

        /**
         * @brief Get the day component.
         *
         * @return The day component.
         */
        unsigned short
        day () const;

        /**
         * @brief Set the day component.
         *
         * @param d The new day component.
         */
        void
        day (unsigned short d);

      protected:
        //@cond

        date ();

        void
        parse (const std::basic_string<C>&);

        //@endcond

      private:
        int year_;
        unsigned short month_;
        unsigned short day_;
      };

      /**
       * @brief %date comparison operator.
       *
       * @return True if the year, month, and day components as well as %time
       * zones are equal, false otherwise.
       */
      template <typename C, typename B>
      bool
      operator== (const date<C, B>&, const date<C, B>&);

      /**
       * @brief %date comparison operator.
       *
       * @return False if the year, month, and day components as well as %time
       * zones are equal, true otherwise.
       */
      template <typename C, typename B>
      bool
      operator!= (const date<C, B>&, const date<C, B>&);


      /**
       * @brief Class corresponding to the XML Schema %time built-in type.
       *
       * The %time class represents hours, minutes, and seconds with an
       * optional %time zone.
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class time: public B, public time_zone
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with the hours, minutes, and
         * seconds components.
         *
         * When this constructor is used, the %time zone is left
         * unspecified.
         *
         * @param hours The hours component.
         * @param minutes The minutes component.
         * @param seconds The seconds component.
         */
        time (unsigned short hours, unsigned short minutes, double seconds);

        /**
         * @brief Initialize an instance with the hours, minutes, and
         * seconds components as well as %time zone.
         *
         * @param hours The hours component.
         * @param minutes The minutes component.
         * @param seconds The seconds component.
         * @param zone_hours The %time zone hours component.
         * @param zone_minutes The %time zone minutes component.
         */
        time (unsigned short hours, unsigned short minutes, double seconds,
              short zone_hours, short zone_minutes);

        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the _clone function instead.
         */
        time (const time& x, flags f = 0, container* c = 0);

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual time*
        _clone (flags f = 0, container* c = 0) const;

        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        time (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        time (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        time (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        time (const std::basic_string<C>& s,
              const xercesc::DOMElement* e,
              flags f = 0,
              container* c = 0);
        //@}

      public:
        /**
         * @brief Get the hours component.
         *
         * @return The hours component.
         */
        unsigned short
        hours () const;

        /**
         * @brief Set the hours component.
         *
         * @param h The new hours component.
         */
        void
        hours (unsigned short h);

        /**
         * @brief Get the minutes component.
         *
         * @return The minutes component.
         */
        unsigned short
        minutes () const;

        /**
         * @brief Set the minutes component.
         *
         * @param m The new minutes component.
         */
        void
        minutes (unsigned short m);

        /**
         * @brief Get the seconds component.
         *
         * @return The seconds component.
         */
        double
        seconds () const;

        /**
         * @brief Set the seconds component.
         *
         * @param s The new seconds component.
         */
        void
        seconds (double s);

      protected:
        //@cond

        time ();

        void
        parse (const std::basic_string<C>&);

        //@endcond

      private:
        unsigned short hours_;
        unsigned short minutes_;
        double seconds_;
      };

      /**
       * @brief %time comparison operator.
       *
       * @return True if the hours, seconds, and minutes components as well
       * as %time zones are equal, false otherwise.
       */
      template <typename C, typename B>
      bool
      operator== (const time<C, B>&, const time<C, B>&);

      /**
       * @brief %time comparison operator.
       *
       * @return False if the hours, seconds, and minutes components as well
       * as %time zones are equal, true otherwise.
       */
      template <typename C, typename B>
      bool
      operator!= (const time<C, B>&, const time<C, B>&);


      /**
       * @brief Class corresponding to the XML Schema dateTime built-in type.
       *
       * The %date_time class represents year, month, day, hours, minutes,
       * and seconds with an optional %time zone.
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class date_time: public B, public time_zone
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with the year, month, day, hours,
         * minutes, and seconds components.
         *
         * When this constructor is used, the %time zone is left
         * unspecified.
         *
         * @param year The year component.
         * @param month The month component.
         * @param day The day component.
         * @param hours The hours component.
         * @param minutes The minutes component.
         * @param seconds The seconds component.
         */
        date_time (int year, unsigned short month, unsigned short day,
                   unsigned short hours, unsigned short minutes,
                   double seconds);

        /**
         * @brief Initialize an instance with the year, month, day, hours,
         * minutes, and seconds components as well as %time zone.
         *
         * @param year The year component.
         * @param month The month component.
         * @param day The day component.
         * @param hours The hours component.
         * @param minutes The minutes component.
         * @param seconds The seconds component.
         * @param zone_hours The %time zone hours component.
         * @param zone_minutes The %time zone minutes component.
         */
        date_time (int year, unsigned short month, unsigned short day,
                   unsigned short hours, unsigned short minutes,
                   double seconds, short zone_hours, short zone_minutes);

        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the _clone function instead.
         */
        date_time (const date_time& x, flags f = 0, container* c = 0);

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual date_time*
        _clone (flags f = 0, container* c = 0) const;

        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        date_time (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        date_time (const xercesc::DOMElement& e,
                   flags f = 0,
                   container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        date_time (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        date_time (const std::basic_string<C>& s,
                   const xercesc::DOMElement* e,
                   flags f = 0,
                   container* c = 0);
        //@}

      public:
        /**
         * @brief Get the year component.
         *
         * @return The year component.
         */
        int
        year () const;

        /**
         * @brief Set the year component.
         *
         * @param y The new year component.
         */
        void
        year (int y);

        /**
         * @brief Get the month component.
         *
         * @return The month component.
         */
        unsigned short
        month () const;

        /**
         * @brief Set the month component.
         *
         * @param m The new month component.
         */
        void
        month (unsigned short m);

        /**
         * @brief Get the day component.
         *
         * @return The day component.
         */
        unsigned short
        day () const;

        /**
         * @brief Set the day component.
         *
         * @param d The new day component.
         */
        void
        day (unsigned short d);

        /**
         * @brief Get the hours component.
         *
         * @return The hours component.
         */
        unsigned short
        hours () const;

        /**
         * @brief Set the hours component.
         *
         * @param h The new hours component.
         */
        void
        hours (unsigned short h);

        /**
         * @brief Get the minutes component.
         *
         * @return The minutes component.
         */
        unsigned short
        minutes () const;

        /**
         * @brief Set the minutes component.
         *
         * @param m The new minutes component.
         */
        void
        minutes (unsigned short m);

        /**
         * @brief Get the seconds component.
         *
         * @return The seconds component.
         */
        double
        seconds () const;

        /**
         * @brief Set the seconds component.
         *
         * @param s The new seconds component.
         */
        void
        seconds (double s);

      protected:
        //@cond

        date_time ();

        void
        parse (const std::basic_string<C>&);

        //@endcond

      private:
        int year_;
        unsigned short month_;
        unsigned short day_;
        unsigned short hours_;
        unsigned short minutes_;
        double seconds_;
      };

      /**
       * @brief %date_time comparison operator.
       *
       * @return True if the year, month, day, hours, seconds, and minutes
       * components as well as %time zones are equal, false otherwise.
       */
      template <typename C, typename B>
      bool
      operator== (const date_time<C, B>&, const date_time<C, B>&);

      /**
       * @brief %date_time comparison operator.
       *
       * @return False if the year, month, day, hours, seconds, and minutes
       * components as well as %time zones are equal, true otherwise.
       */
      template <typename C, typename B>
      bool
      operator!= (const date_time<C, B>&, const date_time<C, B>&);


      /**
       * @brief Class corresponding to the XML Schema %duration built-in type.
       *
       * The %duration class represents a potentially negative %duration in
       * the form of years, months, days, hours, minutes, and seconds.
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class duration: public B
      {
      public:
        /**
         * @name Constructors
         */
        //@{
        /**
         * @brief Initialize a potentially negative instance with the years,
         * months, days, hours, minutes, and seconds components.
         *
         * @param negative A boolean value indicating whether the %duration
         * is negative (true) or positive (false).
         * @param years The years component.
         * @param months The months component.
         * @param days The days component.
         * @param hours The hours component.
         * @param minutes The minutes component.
         * @param seconds The seconds component.
         */
        duration (bool negative,
                  unsigned int years, unsigned int months, unsigned int days,
                  unsigned int hours, unsigned int minutes, double seconds);

        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the _clone function instead.
         */
        duration (const duration& x, flags f = 0, container* c = 0);

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual duration*
        _clone (flags f = 0, container* c = 0) const;

        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        duration (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        duration (const xercesc::DOMElement& e,
                  flags f = 0,
                  container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        duration (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        duration (const std::basic_string<C>& s,
                  const xercesc::DOMElement* e,
                  flags f = 0,
                  container* c = 0);
        //@}

      public:
        /**
         * @brief Determine if %duration is negative.
         *
         * @return True if %duration is negative, false otherwise.
         */
        bool
        negative () const;

        /**
         * @brief Change %duration sign.
         *
         * @param n A boolean value indicating whether %duration is
         * negative (true) or positive (false).
         */
        void
        negative (bool n);

        /**
         * @brief Get the years component.
         *
         * @return The years component.
         */
        unsigned int
        years () const;

        /**
         * @brief Set the years component.
         *
         * @param y The new years component.
         */
        void
        years (unsigned int y);

        /**
         * @brief Get the months component.
         *
         * @return The months component.
         */
        unsigned int
        months () const;

        /**
         * @brief Set the months component.
         *
         * @param m The new months component.
         */
        void
        months (unsigned int m);

        /**
         * @brief Get the days component.
         *
         * @return The days component.
         */
        unsigned int
        days () const;

        /**
         * @brief Set the days component.
         *
         * @param d The new days component.
         */
        void
        days (unsigned int d);

        /**
         * @brief Get the hours component.
         *
         * @return The hours component.
         */
        unsigned int
        hours () const;

        /**
         * @brief Set the hours component.
         *
         * @param h The new hours component.
         */
        void
        hours (unsigned int h);

        /**
         * @brief Get the minutes component.
         *
         * @return The minutes component.
         */
        unsigned int
        minutes () const;

        /**
         * @brief Set the minutes component.
         *
         * @param m The new minutes component.
         */
        void
        minutes (unsigned int m);

        /**
         * @brief Get the seconds component.
         *
         * @return The seconds component.
         */
        double
        seconds () const;

        /**
         * @brief Set the seconds component.
         *
         * @param s The new seconds component.
         */
        void
        seconds (double s);

      protected:
        //@cond

        duration ();

        void
        parse (const std::basic_string<C>&);

        //@endcond

      private:
        bool negative_;
        unsigned int years_;
        unsigned int months_;
        unsigned int days_;
        unsigned int hours_;
        unsigned int minutes_;
        double seconds_;
      };

      /**
       * @brief %duration comparison operator.
       *
       * @return True if the sings as well as years, months, days, hours,
       * seconds, and minutes components are equal, false otherwise.
       */
      template <typename C, typename B>
      bool
      operator== (const duration<C, B>&, const duration<C, B>&);

      /**
       * @brief %duration comparison operator.
       *
       * @return False if the sings as well as years, months, days, hours,
       * seconds, and minutes components are equal, true otherwise.
       */
      template <typename C, typename B>
      bool
      operator!= (const duration<C, B>&, const duration<C, B>&);
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/date-time.ixx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/date-time.txx>

#endif  // XSD_CXX_TREE_DATE_TIME_HXX
