#ifndef LMFIOV_H
#define LMFIOV_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
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
  friend class EcalCondDBInterface; // need permission to write

  LMFIOV();
  LMFIOV(EcalDBConnection *c);
  LMFIOV(const oracle::occi::Environment* env, 
	 const oracle::occi::Connection* conn);
  ~LMFIOV();

  LMFIOV& setStart(const Tm &start);
  LMFIOV& setStop(const Tm &stop);
  LMFIOV& setIOV(const Tm &start, const Tm &stop);
  LMFIOV& setVmin(int vmin);
  LMFIOV& setVmax(int vmax);
  LMFIOV& setVersions(int vmin, int vmax);

  Tm getStart() const;
  Tm getStop() const;
  int getVmin() const;
  int getVmax() const;

  void dump() const;

 private:
  // Methods from LMFUnique
  std::string writeDBSql(Statement *stmt);
  std::string fetchIdSql(Statement *stmt); 
  //  std::string fetchAllSql(Statement *stmt) const;
  std::string setByIDSql(Statement *stmt,
			 int id);
  
  void getParameters(ResultSet *rset);
  void fetchParentIDs() {}
  LMFUnique * createObject() const;
  
  Tm m_iov_start;
  Tm m_iov_stop;
  int m_vmin;
  int m_vmax;

};

#endif
