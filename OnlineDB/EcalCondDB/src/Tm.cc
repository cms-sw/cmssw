#include <ctime>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdio>

#include "OnlineDB/EcalCondDB/interface/Tm.h"

using namespace std;

/**
 * GO: the maximum UNIX time is restricted to INT_MAX, corresponding to
 *     2038-01-19 03:14:07. Take it into account in str()
 * 
 */

std::ostream &operator<<(std::ostream &out, const Tm &t) {
  out << t.str();
  return out;
}

// Default Constructor
Tm::Tm() { this->setNull(); }

// Initialized Constructor
Tm::Tm(struct tm *initTm) { m_tm = *initTm; }

Tm::Tm(uint64_t micros) {
  this->setNull();
  if (micros > PLUS_INF_MICROS) {
    //micros = PLUS_INF_MICROS;
    this->setToCmsNanoTime(micros);
  } else {
    this->setToMicrosTime(micros);
  }
}

// Destructor
Tm::~Tm() {}

struct tm Tm::c_tm() const { return m_tm; }

int Tm::isNull() const {
  if (m_tm.tm_year == 0 && m_tm.tm_mon == 0 && m_tm.tm_mday == 0) {
    return 1;
  } else {
    return 0;
  }
}

void Tm::setNull() {
  m_tm.tm_hour = 0;
  m_tm.tm_isdst = 0;
  m_tm.tm_mday = 0;
  m_tm.tm_min = 0;
  m_tm.tm_mon = 0;
  m_tm.tm_sec = 0;
  m_tm.tm_wday = 0;
  m_tm.tm_yday = 0;
  m_tm.tm_year = 0;
}

string Tm::str() const {
  if (this->isNull()) {
    return "";
  }

  /** 
   *   \brief "One hour shif" fix
   *   Create a temporary dummy object that is in GMT and use it 
   *   to generate the output. This is to avoid the "one hour 
   *   shift" related to the Summer time and the value of 
   *   m_tm.tm_isdst, see [1].  It guaranties that the output
   *   is always in GMT / UTC.
   *   [1] https://hypernews.cern.ch/HyperNews/CMS/get/ecalDB/66.html
   */
  char timebuf[20] = "";
  if (this->microsTime() >= PLUS_INF_MICROS) {
    sprintf(timebuf, "9999-12-12 23:59:59");
  } else {
    Tm dummy_Tm;
    dummy_Tm.setToGMTime(this->microsTime() / 1000000);
    struct tm dummy_tm = dummy_Tm.c_tm();
    strftime(timebuf, 20, "%Y-%m-%d %H:%M:%S", &dummy_tm);
  }
  return string(timebuf);
}

uint64_t Tm::cmsNanoSeconds() const { return microsTime() / 1000000 << 32; }

uint64_t Tm::unixTime() const { return microsTime() / 1000000; }

uint64_t Tm::microsTime() const {
  uint64_t result = 0;
  /*  
  result += (uint64_t)ceil((m_tm.tm_year - 70 ) * 365.25) * 24 * 3600;
  result += (m_tm.tm_yday) * 24 * 3600;
  result += m_tm.tm_hour * 3600;
  result += m_tm.tm_min * 60;
  result += m_tm.tm_sec;
  return (uint64_t) (result * 1000000);
  */

  struct tm time_struct;
  time_struct.tm_year = 1970 - 1900;
  time_struct.tm_mon = 0;
  time_struct.tm_mday = 1;
  time_struct.tm_sec = 0;
  time_struct.tm_min = 0;
  time_struct.tm_hour = 0;
  time_struct.tm_isdst = 0;

  time_t t1970 = mktime(&time_struct);
  tm s = m_tm;
  time_t t_this = mktime(&s);

  double x = difftime(t_this, t1970);
  result = (uint64_t)x * 1000000;

  return result;
}

void Tm::setToCmsNanoTime(uint64_t nanos) { setToMicrosTime((nanos >> 32) * 1000000); }

void Tm::setToMicrosTime(uint64_t micros) {
  time_t t = micros / 1000000;
  if (t >= INT_MAX) {
    t = INT_MAX;
  }
  m_tm = *gmtime(&t);
}

void Tm::setToCurrentLocalTime() {
  time_t t = time(nullptr);
  m_tm = *localtime(&t);
}

void Tm::setToCurrentGMTime() {
  time_t t = time(nullptr);
  m_tm = *gmtime(&t);
}

void Tm::setToLocalTime(time_t t) { m_tm = *localtime(&t); }

void Tm::setToGMTime(time_t t) { m_tm = *gmtime(&t); }

void Tm::setToString(const string s) noexcept(false) {
  sscanf(s.c_str(),
         "%04d-%02d-%02d %02d:%02d:%02d",
         &m_tm.tm_year,
         &m_tm.tm_mon,
         &m_tm.tm_mday,
         &m_tm.tm_hour,
         &m_tm.tm_min,
         &m_tm.tm_sec);

  try {
    if (m_tm.tm_year > 9999 || m_tm.tm_year < 1900) {
      throw(std::runtime_error("Year out of bounds"));
    } else if (m_tm.tm_mon > 12 || m_tm.tm_mon < 1) {
      throw(std::runtime_error("Month out of bounds"));
    } else if (m_tm.tm_mday > 31 || m_tm.tm_mday < 1) {
      throw(std::runtime_error("Day out of bounds"));
    } else if (m_tm.tm_hour > 23 || m_tm.tm_mday < 0) {
      throw(std::runtime_error("Hour out of bounds"));
    } else if (m_tm.tm_min > 59 || m_tm.tm_min < 0) {
      throw(std::runtime_error("Minute out of bounds"));
    } else if (m_tm.tm_sec > 59 || m_tm.tm_sec < 0) {
      throw(std::runtime_error("Day out of bounds"));
    }

    if (m_tm.tm_year >= 2038) {
      // take into account UNIX time limits
      m_tm.tm_year = 2038;
      if (m_tm.tm_mon > 1) {
        m_tm.tm_mon = 1;
      }
      if (m_tm.tm_mday > 19) {
        m_tm.tm_mday = 19;
      }
      if (m_tm.tm_hour > 3) {
        m_tm.tm_hour = 3;
      }
      if (m_tm.tm_min > 14) {
        m_tm.tm_min = 14;
      }
      if (m_tm.tm_sec > 7) {
        m_tm.tm_sec = 7;
      }
    }
    m_tm.tm_year -= 1900;
    m_tm.tm_mon -= 1;
  } catch (std::runtime_error &e) {
    this->setNull();
    string msg("Tm::setToString():  ");
    msg.append(e.what());
    throw(std::runtime_error(msg));
  }
}

void Tm::dumpTm() {
  cout << "=== dumpTm() ===" << endl;
  cout << "tm_year  " << m_tm.tm_year << endl;
  cout << "tm_mon   " << m_tm.tm_mon << endl;
  cout << "tm_mday  " << m_tm.tm_mday << endl;
  cout << "tm_hour  " << m_tm.tm_hour << endl;
  cout << "tm_min   " << m_tm.tm_min << endl;
  cout << "tm_sec   " << m_tm.tm_sec << endl;
  cout << "tm_yday  " << m_tm.tm_yday << endl;
  cout << "tm_wday  " << m_tm.tm_wday << endl;
  cout << "tm_isdst " << m_tm.tm_isdst << endl;
  cout << "================" << endl;
}
