#ifndef LMFSEQVERS_H
#define LMFSEQVERS_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/LMFPrimVers.h"
#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"

/**
 *   LMF sequence version
 */
class LMFSeqVers : public LMFPrimVers {
public:
  friend class LMFRunIOV;  // needs permission to write

  LMFSeqVers();
  LMFSeqVers(EcalDBConnection *c);
  LMFSeqVers(oracle::occi::Environment *env, oracle::occi::Connection *conn);
  ~LMFSeqVers() override;

  // Operators
  inline bool operator==(const LMFSeqVers &t) const { return (getDescription() == t.getDescription()); }
  inline bool operator!=(const LMFSeqVers &t) const { return (getDescription() != t.getDescription()); }

private:
  // Methods from LMFUnique
  std::string fetchIdSql(Statement *stmt) override;
  std::string fetchAllSql(Statement *stmt) const override;
  std::string setByIDSql(Statement *stmt, int id) override;
  void getParameters(ResultSet *rset) override;
  LMFUnique *createObject() const override;
};

#endif
