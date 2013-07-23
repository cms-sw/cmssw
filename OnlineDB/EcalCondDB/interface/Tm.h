// $Id: Tm.h,v 1.6 2011/05/21 06:19:34 organtin Exp $

#ifndef TM_HH
#define TM_HH

#include <stdexcept>
#include <string>
#include <iostream>
#include <time.h>
#include <stdint.h>
#include <limits.h>

// Wrapper class for time.h tm struct
class Tm {
  static const uint64_t NEG_INF_MICROS = 0;
  // GO: maximum UNIX time
  static const uint64_t PLUS_INF_MICROS = (uint64_t)INT_MAX * 1000000;

 public:
  // Default constructor makes a null Tm
  Tm();

  // Initialized constructor
  Tm(struct tm * initTm);

  Tm(uint64_t micros);

  // Destructor
  virtual ~Tm();

  /*
   *  Return a pointer to the tm data structure.
   */
  struct tm c_tm() const;

  /*
   *  Returns true if this Tm is null
   */
  int isNull() const;

  /*
   *  Sets this Tm to null
   */
  void setNull();


  static Tm plusInfinity()
    {
      return Tm(PLUS_INF_MICROS);
    };


  static Tm negInfinity()
    {
      return Tm(NEG_INF_MICROS);
    };

  /*
   *  String representation of Tm in YYYY-MM-DD hh:mm:ss format
   */
  std::string str() const;

  /*
   *  return number of microseconds since Jan 1 1970 and the epoch
   */
  uint64_t unixTime() const;
  uint64_t epoch() const { return unixTime(); };
  uint64_t microsTime() const;

  /*
   *  return the number of nanoseconds packed as a CMS time
   */
  uint64_t cmsNanoSeconds() const;

  /*
   *  Set self to current time
   */
  void setToCurrentLocalTime();
  void setToCurrentGMTime();

  /*
   *  Set using time_t
   */
  void setToLocalTime( time_t t);
  void setToGMTime( time_t t );

  /*
   *  Set using microseconds and CMS times
   */
  void setToMicrosTime(uint64_t micros);
  void setToCmsNanoTime(uint64_t nanos);

  /*
   *  Set to string of format YYYY-MM-DD HH:MM:SS
   */
  void setToString(const std::string s) throw(std::runtime_error);

  void dumpTm();

  inline bool operator<(const Tm &t) const 
    {
      return microsTime() < t.microsTime();
    }

  inline bool operator<=(const Tm &t) const 
    {
      return microsTime() <= t.microsTime();
    }

  inline bool operator==(const Tm &t) const
    {   return (m_tm.tm_hour  == t.m_tm.tm_hour &&
		m_tm.tm_isdst == t.m_tm.tm_isdst &&
		m_tm.tm_mday  == t.m_tm.tm_mday &&
		m_tm.tm_min   == t.m_tm.tm_min &&
		m_tm.tm_mon   == t.m_tm.tm_mon &&
		m_tm.tm_sec   == t.m_tm.tm_sec &&
		m_tm.tm_wday  == t.m_tm.tm_wday &&
		m_tm.tm_yday  == t.m_tm.tm_yday &&
		m_tm.tm_year  == t.m_tm.tm_year); 
    }
				    
  inline bool operator!=(const Tm &t) const { return !(t == *this); }

  Tm& operator-=(int seconds) {
    setToMicrosTime(microsTime() - seconds * 1e6);
    return *this;
  }

  Tm& operator+=(int seconds) {
    setToMicrosTime(microsTime() + seconds * 1e6);
    return *this;
  }

  const Tm operator+(int seconds) {
    Tm ret = *this;
    ret += seconds;
    return ret;
  }

  friend std::ostream& operator<< (std::ostream &out, const Tm &t);

 private:
  struct tm m_tm;
};

#endif
