// This file is adapted from the "chrono" include file of the GNU ISO C++ Library.

// This library is free software; you can redistribute it and/or 
// modify it under the terms of the GNU General Public License 
// as published by the Free Software Foundation; either version 3, 
// or (at your option) any later version.  

#ifndef native_h
#define native_h

// C++ standard headers
#include <chrono>

template <typename _Rep, typename _Period = std::ratio<1>>
struct native_duration;

template <typename _Clock, typename _Dur = typename _Clock::native_duration>
struct native_time_point;

template<typename _Tp>
struct __is_native_duration
  : std::false_type
  { };

template<typename _Rep, typename _Period>
struct __is_native_duration<native_duration<_Rep, _Period>>
  : std::true_type
  { };


namespace std {

  // 20.8.2.3 specialization of common_type (for native_duration)
  template<typename _Rep1, typename _Rep2, typename _Period>
    struct common_type<native_duration<_Rep1, _Period>,
                       native_duration<_Rep2, _Period>>
    {
    private:
      typedef typename common_type<_Rep1, _Rep2>::type      __cr;

    public:
      typedef native_duration<__cr, _Period>                type;
    };

  // 20.8.2.3 specialization of common_type (for native_time_point)
  template<typename _Clock, typename _Dur1, typename _Dur2>
    struct common_type<native_time_point<_Clock, _Dur1>,
                       native_time_point<_Clock, _Dur2>>
    {
    private:
      typedef typename common_type<_Dur1, _Dur2>::type      __ct;

    public:
      typedef chrono::time_point<_Clock, __ct>              type;
    };

} // namespace


// native_duration
template <typename _Rep, typename _Period>
struct native_duration
{
  typedef _Rep                                            rep;
  typedef _Period                                         period;

  // construction / copy / destructor
  constexpr native_duration() : __r() { }

  constexpr native_duration(native_duration const &) = default;

  template <typename _Rep2, typename = typename
         std::enable_if<std::is_convertible<_Rep2, rep>::value and
        (std::chrono::treat_as_floating_point<rep>::value or not std::chrono::treat_as_floating_point<_Rep2>::value)>::type>
  constexpr explicit native_duration(_Rep2 const & __rep) :
    __r(static_cast<rep>(__rep)) 
  { }

  /*
   * FIXME keep this, or force using duration_cast to convert to/from std::chrono:duration ?
  template <typename _FromRep, typename _FromPeriod>
  constexpr explicit native_duration(std::chrono::duration<_FromRep, _FromPeriod> const & __dur) :
    __r(static_cast<rep>(period::from_duration(__dur)))
  { }
  */

  ~native_duration() = default;

  native_duration & operator=(native_duration const &) = default;

  // observer
  constexpr rep
  count() const
  { return __r; }

  // arithmetic
  constexpr native_duration
  operator+() const
  { return *this; }

  constexpr native_duration
  operator-() const
  { return native_duration(-__r); }

  native_duration &
  operator++()
  {
    ++__r;
    return *this;
  }

  native_duration
  operator++(int)
  { return native_duration(__r++); }

  native_duration &
  operator--()
  {
    --__r;
    return *this;
  }

  native_duration
  operator--(int)
  { return native_duration(__r--); }

  native_duration &
  operator+=(native_duration const & __d)
  {
    __r += __d.count();
    return *this;
  }

  native_duration &
  operator-=(native_duration const & __d)
  {
    __r -= __d.count();
    return *this;
  }

  native_duration &
  operator*=(rep const & __rhs)
  {
    __r *= __rhs;
    return *this;
  }

  native_duration &
  operator/=(rep const & __rhs)
  {
    __r /= __rhs;
    return *this;
  }

  // DR 934.
  template <typename _Rep2 = rep>
  typename std::enable_if<not std::chrono::treat_as_floating_point<_Rep2>::value, native_duration &>::type
  operator%=(rep const & __rhs)
  {
    __r %= __rhs;
    return *this;
  }

  template <typename _Rep2 = rep>
  typename std::enable_if<not std::chrono::treat_as_floating_point<_Rep2>::value, native_duration &>::type
  operator%=(native_duration const & __d)
  {
    __r %= __d.count();
    return *this;
  }

  // special values
  static constexpr native_duration
  zero()
  { return native_duration(std::chrono::duration_values<rep>::zero()); }

  static constexpr native_duration
  min()
  { return native_duration(std::chrono::duration_values<rep>::min()); }

  static constexpr native_duration
  max()
  { return native_duration(std::chrono::duration_values<rep>::max()); }

  /*
   * FIXME keep this, or force using duration_cast to convert to/from std::chrono:duration ?
  // convert to chrono::duration
  template <typename _ToRep, typename _ToPeriod>
  operator std::chrono::duration<_ToRep, _ToPeriod>() const
  { 
    return period::template to_duration<_ToRep, _ToPeriod>(__r); 
  }
  */

private:
  rep __r;
};


template <typename _Rep1, typename _Rep2, typename _Period>
constexpr native_duration<typename std::common_type<_Rep1, _Rep2>::type, _Period>
operator+(const native_duration<_Rep1, _Period>& __lhs,
          const native_duration<_Rep2, _Period>& __rhs)
{
  typedef typename std::common_type<_Rep1, _Rep2>::type     __cd;
  typedef native_duration<__cd, _Period>                    __dur;
  return __dur(__cd(__lhs.count()) + __cd(__rhs.count()));
}

template <typename _Rep1, typename _Rep2, typename _Period>
constexpr native_duration<typename std::common_type<_Rep1, _Rep2>::type, _Period>
operator-(const native_duration<_Rep1, _Period>& __lhs,
          const native_duration<_Rep2, _Period>& __rhs)
{
  typedef typename std::common_type<_Rep1, _Rep2>::type     __cd;
  typedef native_duration<__cd, _Period>                    __dur;
  return __dur(__cd(__lhs.count()) - __cd(__rhs.count()));
}

/// specialisation of duration_cast
namespace std {
  namespace chrono {

    template<typename _ToDur, typename _Rep, typename _Period>
    constexpr typename enable_if<__is_duration<_ToDur>::value, _ToDur>::type
    duration_cast(native_duration<_Rep, _Period> const & __d)
    {
      return _Period::template to_duration<typename _ToDur::rep, typename _ToDur::period>(__d.count());
    }

    template<typename _ToDur, typename _Rep, typename _Period>
    constexpr typename enable_if<__is_native_duration<_ToDur>::value, _ToDur>::type
    duration_cast(duration<_Rep, _Period> const & __d)
    {
      return _ToDur(_ToDur::period::template from_duration<_Rep, _Period>(__d));
    }

  } // namespace
} // namespace

/// native_time_point
template <typename _Clock, typename _Dur>
struct native_time_point
{
    typedef _Clock                                          clock;
    typedef _Dur                                            duration;
    typedef typename duration::rep                          rep;
    typedef typename duration::period                       period;

    constexpr native_time_point() : __d(duration::zero())
    { }

    constexpr explicit native_time_point(duration const & __dur) :
      __d(__dur)
    { }

    // conversions
    template <typename _Dur2>
    constexpr native_time_point(native_time_point<clock, _Dur2> const & __t) :
      __d(__t.time_since_epoch())
    { }

    // observer
    constexpr duration
    time_since_epoch() const
    { 
      return __d; 
    }

    // arithmetic
    native_time_point &
    operator+=(duration const & __dur)
    {
      __d += __dur;
      return *this;
    }

    native_time_point &
    operator-=(duration const & __dur)
    {
      __d -= __dur;
      return *this;
    }

    // special values
    static constexpr native_time_point
    min()
    { return native_time_point(duration::min()); }

    static constexpr native_time_point
    max()
    { return native_time_point(duration::max()); }

private:
    duration __d;
};

/*
/// time_point_cast
template <typename _ToDur, typename _Clock, typename _Dur>
constexpr typename std::enable_if<__is_duration<_ToDur>::value,
                             native_time_point<_Clock, _ToDur>>::type
time_point_cast(const native_time_point<_Clock, _Dur>& __t)
{
  typedef native_time_point<_Clock, _ToDur>                   __time_point;
  return __time_point(duration_cast<_ToDur>(__t.time_since_epoch()));
}


template <typename _Clock, typename _Dur1,
       typename _Rep2, typename _Period2>
constexpr native_time_point<_Clock,
  typename std::common_type<_Dur1, duration<_Rep2, _Period2>>::type>
operator+(const native_time_point<_Clock, _Dur1>& __lhs,
          const duration<_Rep2, _Period2>& __rhs)
{
  typedef duration<_Rep2, _Period2>                   __dur2;
  typedef typename std::common_type<_Dur1,__dur2>::type    __ct;
  typedef native_time_point<_Clock, __ct>                     __time_point;
  return __time_point(__lhs.time_since_epoch() + __rhs);
}

template <typename _Rep1, typename _Period1,
       typename _Clock, typename _Dur2>
constexpr native_time_point<_Clock,
  typename std::common_type<duration<_Rep1, _Period1>, _Dur2>::type>
operator+(const duration<_Rep1, _Period1>& __lhs,
          const native_time_point<_Clock, _Dur2>& __rhs)
{ 
  typedef duration<_Rep1, _Period1>                   __dur1;
  typedef typename std::common_type<__dur1,_Dur2>::type    __ct;
  typedef native_time_point<_Clock, __ct>                     __time_point;
  return __time_point(__rhs.time_since_epoch() + __lhs); 
}
*/

template <typename _Clock, typename _Dur1, typename _Rep2, typename _Period2>
constexpr native_time_point<_Clock, typename std::common_type<_Dur1, native_duration<_Rep2, _Period2>>::type>
operator-(const native_time_point<_Clock, _Dur1>& __lhs,
          const native_duration<_Rep2, _Period2>& __rhs)
{ 
  typedef native_duration<_Rep2, _Period2>            __dur2;
  typedef typename std::common_type<_Dur1,__dur2>::type    __ct;
  typedef native_time_point<_Clock, __ct>             __time_point;
  return __time_point(__lhs.time_since_epoch() -__rhs); 
}

template <typename _Clock, typename _Dur1, typename _Rep2, typename _Period2>
constexpr native_time_point<_Clock, typename std::common_type<_Dur1, std::chrono::duration<_Rep2, _Period2>>::type>
operator-(const native_time_point<_Clock, _Dur1>& __lhs,
          const std::chrono::duration<_Rep2, _Period2>& __rhs)
{ 
  typedef std::chrono::duration<_Rep2, _Period2>      __dur2;
  typedef typename std::common_type<_Dur1,__dur2>::type    __ct;
  typedef native_time_point<_Clock, __ct>             __time_point;
  return __time_point(__lhs.time_since_epoch() -__rhs); 
}

template <typename _Clock, typename _Dur1, typename _Dur2>
constexpr typename std::common_type<_Dur1, _Dur2>::type
operator-(const native_time_point<_Clock, _Dur1>& __lhs,
          const native_time_point<_Clock, _Dur2>& __rhs)
{ return __lhs.time_since_epoch() - __rhs.time_since_epoch(); }

/*
template <typename _Clock, typename _Dur1, typename _Dur2>
constexpr bool
operator==(const native_time_point<_Clock, _Dur1>& __lhs,
           const native_time_point<_Clock, _Dur2>& __rhs)
{ return __lhs.time_since_epoch() == __rhs.time_since_epoch(); }

template <typename _Clock, typename _Dur1, typename _Dur2>
constexpr bool
operator!=(const native_time_point<_Clock, _Dur1>& __lhs,
           const native_time_point<_Clock, _Dur2>& __rhs)
{ return !(__lhs == __rhs); }

template <typename _Clock, typename _Dur1, typename _Dur2>
constexpr bool
operator<(const native_time_point<_Clock, _Dur1>& __lhs,
          const native_time_point<_Clock, _Dur2>& __rhs)
{ return  __lhs.time_since_epoch() < __rhs.time_since_epoch(); }

template <typename _Clock, typename _Dur1, typename _Dur2>
constexpr bool
operator<=(const native_time_point<_Clock, _Dur1>& __lhs,
           const native_time_point<_Clock, _Dur2>& __rhs)
{ return !(__rhs < __lhs); }

template <typename _Clock, typename _Dur1, typename _Dur2>
constexpr bool
operator>(const native_time_point<_Clock, _Dur1>& __lhs,
          const native_time_point<_Clock, _Dur2>& __rhs)
{ return __rhs < __lhs; }

template <typename _Clock, typename _Dur1, typename _Dur2>
constexpr bool
operator>=(const native_time_point<_Clock, _Dur1>& __lhs,
           const native_time_point<_Clock, _Dur2>& __rhs)
{ return !(__lhs < __rhs); }
*/

#endif //native_h
