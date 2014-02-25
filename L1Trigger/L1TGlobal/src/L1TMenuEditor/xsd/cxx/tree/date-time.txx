// file      : xsd/cxx/tree/date-time.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // gday
      //
      template <typename C, typename B>
      gday<C, B>* gday<C, B>::
      _clone (flags f, container* c) const
      {
        return new gday (*this, f, c);
      }

      // gmonth
      //
      template <typename C, typename B>
      gmonth<C, B>* gmonth<C, B>::
      _clone (flags f, container* c) const
      {
        return new gmonth (*this, f, c);
      }

      // gyear
      //
      template <typename C, typename B>
      gyear<C, B>* gyear<C, B>::
      _clone (flags f, container* c) const
      {
        return new gyear (*this, f, c);
      }

      // gmonth_day
      //
      template <typename C, typename B>
      gmonth_day<C, B>* gmonth_day<C, B>::
      _clone (flags f, container* c) const
      {
        return new gmonth_day (*this, f, c);
      }

      // gyear_month
      //
      template <typename C, typename B>
      gyear_month<C, B>* gyear_month<C, B>::
      _clone (flags f, container* c) const
      {
        return new gyear_month (*this, f, c);
      }

      // date
      //
      template <typename C, typename B>
      date<C, B>* date<C, B>::
      _clone (flags f, container* c) const
      {
        return new date (*this, f, c);
      }

      // time
      //
      template <typename C, typename B>
      time<C, B>* time<C, B>::
      _clone (flags f, container* c) const
      {
        return new time (*this, f, c);
      }

      // date_time
      //
      template <typename C, typename B>
      date_time<C, B>* date_time<C, B>::
      _clone (flags f, container* c) const
      {
        return new date_time (*this, f, c);
      }

      // duration
      //
      template <typename C, typename B>
      duration<C, B>* duration<C, B>::
      _clone (flags f, container* c) const
      {
        return new duration (*this, f, c);
      }
    }
  }
}
