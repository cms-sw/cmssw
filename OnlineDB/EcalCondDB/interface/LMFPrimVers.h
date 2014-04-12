#ifndef LMFPRIMVERS_H
#define LMFPRIMVERS_H

#include <string>
#include <stdexcept>

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
*/

#include "OnlineDB/EcalCondDB/interface/LMFUnique.h"
#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"

/**
 *   LMF version
 *
 *   Versions cannot be written into the database using these classes,
 *   but only via the administration shell. To insert a new version:
 *   INSERT INTO <TABLE_NAME> VALUES (<VERS>, DEFAULT, <DESCR>);
 *
 */
class LMFPrimVers : public LMFUnique {
 public:
  friend class LMFRunIOV;  // needs permission to write

  LMFPrimVers();
  LMFPrimVers(EcalDBConnection *c);
  LMFPrimVers(oracle::occi::Environment* env,
	      oracle::occi::Connection* conn);
  ~LMFPrimVers();

  // Methods for user data
  int getVersion() const { return m_ID; }
  std::string getDescription() const { return getString("description"); }
  void setVersion(int v) { m_ID = v; }
  void setDescription(const std::string &s) { setString("description", s); }

  // Operators
  inline bool operator==(const LMFPrimVers &t) const { 
    return (getID() == t.getID());
  }
  inline bool operator!=(const LMFPrimVers &t) const { 
    return (getID() != t.getID());
  }

 private:
  // Methods from LMFUnique
  std::string fetchIdSql(Statement *stmt);
  std::string fetchAllSql(Statement *stmt) const;
  std::string setByIDSql(Statement *stmt, int id);
  void getParameters(ResultSet *rset);
  LMFUnique *createObject() const;
};

#endif
