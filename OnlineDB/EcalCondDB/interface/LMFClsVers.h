#ifndef LMFCLSVERS_H
#define LMFCLSVERS_H

/*
 Copyright (c) Giovanni.Organtini@roma1.infn.it 2010
 */

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/LMFPrimVers.h"
#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"

/**
 *   LMF Correction version
 */
class LMFClsVers : public LMFPrimVers {
 public:
  friend class LMFRunIOV;  // needs permission to write

  LMFClsVers();
  LMFClsVers(EcalDBConnection *c);
  LMFClsVers(oracle::occi::Environment* env,
	     oracle::occi::Connection* conn);
  ~LMFClsVers();

  // Operators
  inline bool operator==(const LMFClsVers &t) const { 
    return (getID()== t.getID());
  }
  inline bool operator!=(const LMFClsVers &t) const { 
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
