#ifndef LMFCORRVERS_H
#define LMFCORRVERS_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/LMFPrimVers.h"
#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"

/**
 *   LMF Correction version
 */
class LMFCorrVers : public LMFPrimVers {
public:
  friend class LMFRunIOV;  // needs permission to write

  LMFCorrVers();
  LMFCorrVers(EcalDBConnection *c);
  LMFCorrVers(oracle::occi::Environment *env, oracle::occi::Connection *conn);
  ~LMFCorrVers() override;

  // Operators
  inline bool operator==(const LMFCorrVers &t) const { return (getID() == t.getID()); }
  inline bool operator!=(const LMFCorrVers &t) const { return (getID() != t.getID()); }

private:
  // Methods from LMFUnique
  std::string fetchIdSql(Statement *stmt) override;
  std::string fetchAllSql(Statement *stmt) const override;
  std::string setByIDSql(Statement *stmt, int id) override;
  void getParameters(ResultSet *rset) override;
  LMFUnique *createObject() const override;
};

#endif
