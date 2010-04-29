#ifndef LMFIOV_H
#define LMFIOV_H

/*
 Copyright (c) Giovanni.Organtini@roma1.infn.it 2010
 */

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/LMFUnique.h"
#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"

/**
 *   LMF IOV
 */
class LMFIOV : public LMFUnique {
 public:
  friend class LMFRunIOV;  // needs permission to write

  LMFIOV();
  LMFIOV(EcalDBConnection *c);
  LMFIOV(const oracle::occi::Environment* env, 
	 const oracle::occi::Connection* conn);
  ~LMFIOV();

  void dump() const;

  // Operators
  inline bool operator==(const LMFIOV &t) const { 
    return ((m_iov_start == t.m_iov_start) &&
	    (m_iov_stop  == t.m_iov_stop) &&
	    (m_vmin == t.m_vmin) &&
	    (m_vmax == t.m_vmax));
  }
  inline bool operator!=(const LMFIOV &t) const { 
    return ((m_iov_start != t.m_iov_start) ||
	    (m_iov_stop  != t.m_iov_stop) ||
	    (m_vmin != t.m_vmin) ||
	    (m_vmax != t.m_vmax));
  }

 private:
  // Methods from LMFUnique
  std::string fetchIdSql(Statement *stmt);
  std::string setByIDSql(Statement *stmt, int id);
  void getParameters(ResultSet *rset);
  //  LMFUnique *createObject();

  Tm m_iov_start;
  Tm m_iov_stop;
  int m_vmin;
  int m_vmax;

};

#endif
