#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace oracle::occi;

DateHandler::DateHandler(Environment* env, Connection* conn) {
  m_env = env;
  m_conn = conn;

  PLUS_INF_DATE = maxDate();
  PLUS_INF = dateToTm(PLUS_INF_DATE);
  NEG_INF_DATE = minDate();
  NEG_INF = dateToTm(NEG_INF_DATE);
}

DateHandler::~DateHandler() {}

Date DateHandler::tmToDate(const Tm& inTm) const {
  if (inTm.isNull()) {
    return Date();
  } else {
    struct tm ctm = inTm.c_tm();
    return Date(m_env, ctm.tm_year + 1900, ctm.tm_mon + 1, ctm.tm_mday, ctm.tm_hour, ctm.tm_min, ctm.tm_sec);
  }
}

Tm DateHandler::dateToTm(Date& date) const {
  if (date.isNull()) {
    return Tm();
  }

  int year;
  unsigned int mon;   // month
  unsigned int mday;  // day of month
  unsigned int hour;
  unsigned int min;  // minute
  unsigned int sec;  // second

  date.getDate(year, mon, mday, hour, min, sec);

  // work on the provided tm
  struct tm retTm;
  retTm.tm_year = year - 1900;
  retTm.tm_mon = mon - 1;
  retTm.tm_mday = mday;
  retTm.tm_hour = hour;
  retTm.tm_min = min;
  retTm.tm_sec = sec;
  retTm.tm_isdst = 0;
  retTm.tm_wday = 0;
  retTm.tm_yday = 0;

  mktime(&retTm);  // calculates tm_wday and tm_yday

  return Tm(&retTm);
}
