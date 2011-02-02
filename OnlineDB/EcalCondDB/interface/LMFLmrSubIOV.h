#ifndef LMFLMRSUBIOV_H
#define LMFLMRSUBIOV_H

/*
 Copyright (c) Giovanni.Organtini@roma1.infn.it 2010
 */

#include <map>

#include "OnlineDB/EcalCondDB/interface/LMFUnique.h"
#include "OnlineDB/EcalCondDB/interface/LMFIOV.h"

class LMFLmrSubIOV : public LMFUnique {
 public:
  friend class EcalCondDBInterface;

  LMFLmrSubIOV();
  LMFLmrSubIOV(EcalDBConnection *c);
  LMFLmrSubIOV(oracle::occi::Environment* env,
	       oracle::occi::Connection* conn);
  ~LMFLmrSubIOV();

  LMFLmrSubIOV& setLMFIOV(const LMFIOV &iov);
  LMFLmrSubIOV& setLMFIOV(int i);
  LMFLmrSubIOV& setTimes(Tm t1, Tm t2, Tm t3);
  LMFLmrSubIOV& setTimes(std::vector<Tm> t);
  LMFLmrSubIOV& setTimes(Tm *t); // array of three components
  
  int getLMFIOVID() const {
    return m_lmfIOV;
  };
  LMFIOV getLMFIOV() const {
    LMFIOV iov(m_env, m_conn);
    iov.setByID(m_lmfIOV);
    return iov;
  };
  inline void getTimes(Tm *t1, Tm *t2, Tm *t3) const {
    *t1 = m_t[0];
    *t2 = m_t[1];
    *t3 = m_t[2];
  }
  std::vector<Tm> getTimes() const {
    std::vector<Tm> v;
    v.push_back(m_t[0]);
    v.push_back(m_t[1]);
    v.push_back(m_t[2]);
    return v;
  }
  inline void getTimes(Tm *t) const {
    t[0] = m_t[0];
    t[1] = m_t[1];
    t[2] = m_t[2];
  }
  inline Tm getTime(int i) {
    Tm t;
    if ((i > 0) && (i < 4)) {
      t = m_t[i - 1];
    }
    return t;
  }
  inline Tm getT1() {
    return m_t[0];
  }
  inline Tm getT2() {
    return m_t[1];
  }
  inline Tm getT3() {
    return m_t[2];
  }

  std::list<int> getIOVIDsLaterThan(const Tm &t)
    throw(std::runtime_error);
  std::list<int> getIOVIDsLaterThan(const Tm &tmin, const Tm &tmax)
    throw(std::runtime_error);
  std::list<int> getIOVIDsLaterThan(const Tm &t, int howMany)
    throw(std::runtime_error);
  std::list<int> getIOVIDsLaterThan(const Tm &t, const Tm &tmax,
				    int howMany)
    throw(std::runtime_error);

  // Operators
  inline bool operator==(const LMFLmrSubIOV &m) const
    {
      return ( (m_lmfIOV   == m.m_lmfIOV) &&
	       (m_t[0]    == m.m_t[0]) &&
	       (m_t[1]    == m.m_t[1]) && 
	       (m_t[2]    == m.m_t[2]));
    }

  inline bool operator!=(const LMFLmrSubIOV &m) const { return !(*this == m); }

  std::list<LMFLmrSubIOV> fetchByLMFIOV(const LMFIOV &iov);

 private:
  int m_lmfIOV;
  Tm m_t[3];

  std::string fetchIdSql(Statement *stmt);
  std::string setByIDSql(Statement *stmt, int id);
  std::string writeDBSql(Statement *stmt);
  void getParameters(ResultSet *rset);
  void init();
};

#endif
