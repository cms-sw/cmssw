#ifndef DATEHANDLER_H
#define DATEHANDLER_H

#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

class DateHandler {
public:
  DateHandler(oracle::occi::Environment* env, oracle::occi::Connection* conn);
  ~DateHandler();

  inline Tm getNegInfTm() const { return NEG_INF; }
  inline Tm getPlusInfTm() const { return PLUS_INF; }
  inline oracle::occi::Date getNegInfDate() const { return NEG_INF_DATE; }
  inline oracle::occi::Date getPlusInfDate() const { return PLUS_INF_DATE; }

  /**
   *  Get the current system date
   */
  inline oracle::occi::Date getCurrentDate() { return oracle::occi::Date(oracle::occi::Date::getSystemDate(m_env)); }

  /**
   *  The minimum oracle Date
   */
  inline oracle::occi::Date minDate() { return oracle::occi::Date(m_env, 1970, 1, 1, 0, 0, 0); }

  /**
   *  The maximum oracle Date
   */
  inline oracle::occi::Date maxDate() { return oracle::occi::Date(m_env, 9999, 12, 31, 23, 59, 59); }

  /**
   *  Translate a Tm object to a oracle Date object
   */
  oracle::occi::Date tmToDate(const Tm& inTm) const;

  /**
   *  Translate an oracle Date object to a Tm object
   */
  Tm dateToTm(oracle::occi::Date& date) const;

  DateHandler() = delete;  // hide the default constructor

private:
  oracle::occi::Connection* m_conn;
  oracle::occi::Environment* m_env;

  Tm PLUS_INF;
  Tm NEG_INF;
  oracle::occi::Date PLUS_INF_DATE;
  oracle::occi::Date NEG_INF_DATE;
};

#endif
